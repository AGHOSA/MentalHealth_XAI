import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ==================================================
# Configuration
# ==================================================
SEED = 42
MODEL_NAME = "bert-base-uncased"

DATA_PATH = Path("data/pirina_reddit_preprocessed.csv")
OUTPUT_DIR = Path("results/optimization/bert_ga")

MAX_LENGTH = 128
BATCH_SIZE = 8

# Journal-inspired GA configuration
POPULATION_SIZE = 10
N_GENERATIONS = 5
MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.90
ELITE_SIZE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# Reproducibility
# ==================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================================================
# Validate dataset
# ==================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {"text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain columns: {required_columns}"
    )

df = df[["text", "label"]].dropna().copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()
df["label"] = df["label"].astype(int)

if set(df["label"].unique()) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

# ==================================================
# Same 80/10/10 split
# ==================================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"],
)

print(
    f"Dataset: {len(df)} | "
    f"Train: {len(train_df)} | "
    f"Validation: {len(val_df)} | "
    f"Reserved test: {len(test_df)}"
)

# ==================================================
# Tokenization
# ==================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def make_dataset(frame):
    dataset = Dataset.from_pandas(
        frame[["text", "label"]].reset_index(drop=True)
    )

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
    )

    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")

    return dataset


train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)

# ==================================================
# Metrics
# ==================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


# ==================================================
# GA Search Space
# ==================================================
# Each chromosome contains:
#
# [learning_rate,
#  weight_decay,
#  warmup_ratio,
#  epochs]
#
# Values are represented as normalized numbers [0, 1]
# and decoded into actual hyperparameters.
# ==================================================

def random_individual():
    return [
        random.random(),  # learning rate
        random.random(),  # weight decay
        random.random(),  # warmup ratio
        random.random(),  # epochs
    ]


def decode(individual):
    # Learning rate: 5e-6 -> 5e-5 (log scale)
    log_min = np.log10(5e-6)
    log_max = np.log10(5e-5)

    learning_rate = 10 ** (
        log_min
        + individual[0] * (log_max - log_min)
    )

    # Weight decay: 0 -> 0.1
    weight_decay = individual[1] * 0.1

    # Warmup ratio: 0 -> 0.15
    warmup_ratio = individual[2] * 0.15

    # Epochs: 2 -> 5
    epochs = int(
        round(2 + individual[3] * 3)
    )

    # Convert warmup ratio to warmup steps.
    # Training set = 1472 samples, batch size = 8.
    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * epochs

    warmup_steps = int(
        round(warmup_ratio * total_training_steps)
    )

    return {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_ratio": float(warmup_ratio),
        "warmup_steps": warmup_steps,
        "epochs": epochs,
    }


# ==================================================
# Fitness Evaluation
# ==================================================
evaluation_cache = {}


def evaluate_individual(individual, generation, index):
    params = decode(individual)

    cache_key = tuple(
        round(float(x), 8)
        for x in individual
    )

    if cache_key in evaluation_cache:
        return evaluation_cache[cache_key]

    trial_dir = (
        OUTPUT_DIR
        / "generations"
        / f"generation_{generation:02d}"
        / f"individual_{index:02d}"
    )

    trial_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(
        f"Generation {generation} | "
        f"Individual {index}"
    )
    print("=" * 60)

    print(json.dumps(params, indent=2))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "non_depression",
            1: "depression",
        },
        label2id={
            "non_depression": 0,
            "depression": 1,
        },
    )

    args = TrainingArguments(
        output_dir=str(trial_dir),
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_steps=params["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    try:
        trainer.train()

        metrics = trainer.evaluate()

        fitness = float(
            metrics["eval_macro_f1"]
        )

        result = {
            "fitness_macro_f1": fitness,
            "validation_accuracy": float(
                metrics["eval_accuracy"]
            ),
            "validation_macro_precision": float(
                metrics["eval_macro_precision"]
            ),
            "validation_macro_recall": float(
                metrics["eval_macro_recall"]
            ),
            "parameters": params,
        }

        print(
            f"Validation Accuracy: "
            f"{result['validation_accuracy']:.4f}"
        )

        print(
            f"Validation Macro-F1: "
            f"{result['fitness_macro_f1']:.4f}"
        )

        evaluation_cache[cache_key] = result

        return result

    finally:
        del trainer
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================================================
# Tournament Selection
# ==================================================
def tournament_selection(
    population,
    fitnesses,
    tournament_size=3,
):
    selected_indices = random.sample(
        range(len(population)),
        tournament_size,
    )

    winner = max(
        selected_indices,
        key=lambda i: fitnesses[i],
    )

    return population[winner].copy()


# ==================================================
# Crossover
# ==================================================
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    point = random.randint(
        1,
        len(parent1) - 1,
    )

    child1 = (
        parent1[:point]
        + parent2[point:]
    )

    child2 = (
        parent2[:point]
        + parent1[point:]
    )

    return child1, child2


# ==================================================
# Mutation
# ==================================================
def mutate(individual):
    child = individual.copy()

    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] = random.random()

    return child


# ==================================================
# Main GA
# ==================================================
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

population = [
    random_individual()
    for _ in range(POPULATION_SIZE)
]

all_results = []

best_individual = None
best_result = None

for generation in range(
    1,
    N_GENERATIONS + 1,
):
    print("\n")
    print("#" * 70)
    print(
        f"GENERATION {generation}/{N_GENERATIONS}"
    )
    print("#" * 70)

    generation_results = []
    fitnesses = []

    for index, individual in enumerate(
        population
    ):
        result = evaluate_individual(
            individual,
            generation,
            index,
        )

        fitness = result[
            "fitness_macro_f1"
        ]

        fitnesses.append(fitness)

        row = {
            "generation": generation,
            "individual": index,
            "fitness_macro_f1": fitness,
            "validation_accuracy": result[
                "validation_accuracy"
            ],
            "validation_macro_precision": result[
                "validation_macro_precision"
            ],
            "validation_macro_recall": result[
                "validation_macro_recall"
            ],
            **result["parameters"],
        }

        generation_results.append(row)
        all_results.append(row)

        if (
            best_result is None
            or fitness
            > best_result[
                "fitness_macro_f1"
            ]
        ):
            best_result = result
            best_individual = individual.copy()

    # Save generation results
    generation_file = (
        OUTPUT_DIR
        / f"generation_{generation:02d}_results.csv"
    )

    pd.DataFrame(
        generation_results
    ).to_csv(
        generation_file,
        index=False,
    )

    # ==================================================
    # Elitism
    # ==================================================
    ranked_indices = sorted(
        range(len(population)),
        key=lambda i: fitnesses[i],
        reverse=True,
    )

    new_population = []

    for i in ranked_indices[
        :ELITE_SIZE
    ]:
        new_population.append(
            population[i].copy()
        )

    # ==================================================
    # Create next generation
    # ==================================================
    while len(new_population) < POPULATION_SIZE:

        parent1 = tournament_selection(
            population,
            fitnesses,
        )

        parent2 = tournament_selection(
            population,
            fitnesses,
        )

        child1, child2 = crossover(
            parent1,
            parent2,
        )

        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(child1)

        if (
            len(new_population)
            < POPULATION_SIZE
        ):
            new_population.append(child2)

    population = new_population

    print("\nGeneration summary:")
    print(
        f"Best validation Macro-F1: "
        f"{max(fitnesses):.4f}"
    )


# ==================================================
# Save final GA results
# ==================================================
results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_DIR / "ga_all_results.csv",
    index=False,
)

best_parameters = {
    "optimization_method": (
        "Genetic Algorithm"
    ),
    "population_size": POPULATION_SIZE,
    "generations": N_GENERATIONS,
    "mutation_rate": MUTATION_RATE,
    "crossover_rate": CROSSOVER_RATE,
    "elite_size": ELITE_SIZE,
    "objective": "validation_macro_f1",
    "pretrained_model": MODEL_NAME,
    "dataset_size": len(df),
    "train_size": len(train_df),
    "validation_size": len(val_df),
    "reserved_test_size": len(test_df),
    "best_validation_macro_f1": float(
        best_result[
            "fitness_macro_f1"
        ]
    ),
    "best_validation_accuracy": float(
        best_result[
            "validation_accuracy"
        ]
    ),
    "best_parameters": best_result[
        "parameters"
    ],
}

(OUTPUT_DIR / "best_parameters.json").write_text(
    json.dumps(
        best_parameters,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n")
print("=" * 70)
print("GENETIC ALGORITHM OPTIMIZATION COMPLETED")
print("=" * 70)

print(
    json.dumps(
        best_parameters,
        indent=2,
    )
)

print(
    "\nIMPORTANT: The reserved test set "
    "was NOT used during GA optimization."
)

print(
    f"\nResults saved in: {OUTPUT_DIR}"
)