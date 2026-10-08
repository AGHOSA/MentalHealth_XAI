import json
import gc
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


# ============================================================
# Differential Evolution (DE) optimization for BERT
# ============================================================
# Purpose:
#   Optimize BERT training hyperparameters + decision threshold
#   using ONLY the validation set.
#
# Final test set remains untouched until the very end.
#
# Baseline protocol preserved:
#   Model: bert-base-uncased
#   Max length: 128
#   Batch size: 8
#   Seed: 42
#   Split: 80% train / 10% validation / 10% test
#
# Search variables:
#   x[0] = learning rate (log10 scale)
#   x[1] = weight decay
#   x[2] = warmup ratio
#   x[3] = epochs (rounded to integer)
#   x[4] = decision threshold for class 1 (depression)
#
# NOTE:
# This is a research experiment. A higher result is NOT guaranteed.
# Do not change the test set or optimize on test metrics.
# ============================================================


SEED = 42
MODEL_NAME = "bert-base-uncased"

DATA_PATH = Path("data/pirina_reddit_preprocessed.csv")
OUTPUT_ROOT = Path("results/models/bert_de")
BEST_MODEL_DIR = OUTPUT_ROOT / "best_model"
TRIAL_ROOT = OUTPUT_ROOT / "trials"
RESULT_JSON = OUTPUT_ROOT / "optimization_results.json"
TRIAL_CSV = OUTPUT_ROOT / "trial_results.csv"

MAX_LENGTH = 128
BATCH_SIZE = 8

# Laptop-friendly DE settings.
# 4 individuals x 2 generations = 12 training runs total.
POPULATION_SIZE = 4
GENERATIONS = 2

# Differential Evolution parameters.
F = 0.7       # mutation factor
CR = 0.8      # crossover probability

# Search bounds.
# learning rate is represented by log10(lr)
BOUNDS = [
    (-5.0, -4.2),   # lr => about 1e-5 to 6.3e-5
    (0.0, 0.10),    # weight decay
    (0.0, 0.10),    # warmup ratio
    (2.0, 5.0),     # epochs
    (0.40, 0.70),   # decision threshold
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_params(x):
    lr = 10 ** float(x[0])
    weight_decay = float(x[1])
    warmup_ratio = float(x[2])
    epochs = int(round(float(x[3])))
    threshold = float(x[4])

    epochs = max(2, min(5, epochs))
    threshold = max(0.40, min(0.70, threshold))

    return {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "epochs": epochs,
        "threshold": threshold,
    }


def make_dataset(frame, tokenizer):
    dataset = Dataset.from_pandas(
        frame[["text", "label"]].reset_index(drop=True)
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset


def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "non_depression", 1: "depression"},
        label2id={"non_depression": 0, "depression": 1},
    )


def cleanup_dir(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def compute_validation_metrics(trainer, val_dataset, threshold):
    """
    Returns validation metrics using a tunable class-1 probability threshold.
    """
    pred_output = trainer.predict(val_dataset)

    logits = pred_output.predictions
    labels = pred_output.label_ids

    # Stable sigmoid for the positive/depression logit difference.
    logit_diff = logits[:, 1] - logits[:, 0]
    probs_class1 = 1.0 / (1.0 + np.exp(-np.clip(logit_diff, -50, 50)))

    predictions = (probs_class1 >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "predictions": predictions,
        "labels": labels,
    }


def objective(x, trial_id, train_dataset, val_dataset):
    """
    DE objective:
      maximize validation accuracy.
    We minimize negative validation accuracy for DE.
    """
    params = decode_params(x)

    trial_dir = TRIAL_ROOT / f"trial_{trial_id:02d}"
    cleanup_dir(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"TRIAL {trial_id}")
    print("=" * 78)
    print(json.dumps(params, indent=2))

    set_seed(SEED)

    model = build_model()

    training_args = TrainingArguments(
        output_dir=str(trial_dir),
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_steps=int(
    (len(train_dataset) // BATCH_SIZE)
    * params["epochs"]
    * params["warmup_ratio"]
),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    def compute_trainer_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    val_metrics = compute_validation_metrics(
        trainer,
        val_dataset,
        params["threshold"],
    )

    record = {
        "trial": trial_id,
        **params,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_macro_f1": val_metrics["macro_f1"],
    }

    print("\nValidation result:")
    print(json.dumps(record, indent=2))

    del trainer
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cleanup_dir(trial_dir)

    return record


def differential_evolution(train_dataset, val_dataset):
    rng = np.random.default_rng(SEED)

    dimension = len(BOUNDS)

    # Initial population.
    population = np.array(
        [
            [
                rng.uniform(low, high)
                for (low, high) in BOUNDS
            ]
            for _ in range(POPULATION_SIZE)
        ],
        dtype=float,
    )

    history = []
    fitness = np.empty(POPULATION_SIZE, dtype=float)

    # Initial evaluation.
    for i in range(POPULATION_SIZE):
        result = objective(
            population[i],
            trial_id=i + 1,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        history.append(result)
        fitness[i] = -result["val_accuracy"]

    next_trial = POPULATION_SIZE + 1

    # DE generations.
    for generation in range(1, GENERATIONS + 1):
        print("\n" + "#" * 78)
        print(f"GENERATION {generation}/{GENERATIONS}")
        print("#" * 78)

        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(POPULATION_SIZE):
            candidates = [j for j in range(POPULATION_SIZE) if j != i]
            a, b, c = rng.choice(candidates, size=3, replace=False)

            mutant = (
                population[a]
                + F * (population[b] - population[c])
            )

            # Clip mutant to allowed bounds.
            for d, (low, high) in enumerate(BOUNDS):
                mutant[d] = np.clip(mutant[d], low, high)

            # Binomial crossover.
            trial_vector = population[i].copy()
            forced_index = rng.integers(0, dimension)

            for d in range(dimension):
                if rng.random() < CR or d == forced_index:
                    trial_vector[d] = mutant[d]

            result = objective(
                trial_vector,
                trial_id=next_trial,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
            next_trial += 1
            history.append(result)

            trial_fitness = -result["val_accuracy"]

            if trial_fitness <= fitness[i]:
                new_population[i] = trial_vector
                new_fitness[i] = trial_fitness
                print(f"Trial accepted for individual {i}.")
            else:
                print(f"Trial rejected for individual {i}.")

        population = new_population
        fitness = new_fitness

        best_index = int(np.argmin(fitness))
        best = decode_params(population[best_index])

        print("\nBest after generation:")
        print(json.dumps(best, indent=2))
        print(
            f"Best validation accuracy: "
            f"{(-fitness[best_index]) * 100:.2f}%"
        )

    best_index = int(np.argmin(fitness))
    best_vector = population[best_index]
    best_params = decode_params(best_vector)

    # Find best recorded result.
    best_record = max(
        history,
        key=lambda r: r["val_accuracy"],
    )

    return best_params, best_record, history


def final_test_evaluation(best_params, train_dataset, val_dataset, test_dataset):
    """
    Retrain exactly once with the selected parameters and evaluate on the
    untouched test set. The test set is NOT used during DE.
    """
    print("\n" + "=" * 78)
    print("FINAL OPTIMIZED BERT TRAINING")
    print("=" * 78)
    print(json.dumps(best_params, indent=2))

    cleanup_dir(BEST_MODEL_DIR)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    model = build_model()

    args = TrainingArguments(
        output_dir=str(BEST_MODEL_DIR / "checkpoints"),
        num_train_epochs=best_params["epochs"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    def compute_trainer_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_trainer_metrics,
    )

    trainer.train()

    # Predict test set.
    prediction_output = trainer.predict(test_dataset)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids

    logit_diff = logits[:, 1] - logits[:, 0]
    probs_class1 = 1.0 / (
        1.0 + np.exp(-np.clip(logit_diff, -50, 50))
    )

    threshold = best_params["threshold"]
    predictions = (probs_class1 >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(labels, predictions).tolist()

    final_metrics = {
        "model": MODEL_NAME,
        "optimization": "Differential Evolution",
        "fitness": "validation_accuracy",
        "seed": SEED,
        "dataset_size": int(len(labels) + len(train_dataset) + len(val_dataset)),
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "test_size": len(test_dataset),
        "learning_rate": best_params["learning_rate"],
        "weight_decay": best_params["weight_decay"],
        "warmup_ratio": best_params["warmup_ratio"],
        "epochs": best_params["epochs"],
        "decision_threshold": best_params["threshold"],
        "test_accuracy": float(accuracy),
        "test_macro_precision": float(precision),
        "test_macro_recall": float(recall),
        "test_macro_f1": float(f1),
        "confusion_matrix": cm,
    }

    print("\nFINAL TEST RESULT")
    print(json.dumps(final_metrics, indent=2))

    trainer.save_model(str(BEST_MODEL_DIR / "final_model"))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR / "final_model"))

    return final_metrics


def main():
    set_seed(SEED)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BERT DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CPU mode. This experiment may take a long time.")

    # Load data.
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 and 1.")

    print(f"Dataset size: {len(df)}")
    print(df["label"].value_counts().sort_index())

    # Exact same split protocol as baseline.
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
        f"Split sizes -> Train: {len(train_df)}, "
        f"Validation: {len(val_df)}, "
        f"Test: {len(test_df)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = make_dataset(train_df, tokenizer)
    val_dataset = make_dataset(val_df, tokenizer)
    test_dataset = make_dataset(test_df, tokenizer)

    best_params, best_record, history = differential_evolution(
        train_dataset,
        val_dataset,
    )

    print("\n" + "=" * 78)
    print("OPTIMIZATION COMPLETE")
    print("=" * 78)
    print("Best parameters:")
    print(json.dumps(best_params, indent=2))

    print("\nBest validation record:")
    print(json.dumps(best_record, indent=2))

    # Save trial table.
    history_df = pd.DataFrame(history)
    history_df.to_csv(TRIAL_CSV, index=False)

    final_metrics = final_test_evaluation(
        best_params,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    result = {
        "best_parameters": best_params,
        "best_validation_record": best_record,
        "final_test_metrics": final_metrics,
        "baseline_test_accuracy_reference": 0.9402173913043478,
        "test_accuracy_change_vs_baseline_percentage_points": (
            (final_metrics["test_accuracy"] - 0.9402173913043478) * 100
        ),
    }

    RESULT_JSON.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)
    print(f"Baseline BERT accuracy: 94.02%")
    print(
        f"Optimized BERT accuracy: "
        f"{final_metrics['test_accuracy'] * 100:.2f}%"
    )
    print(
        f"Change: "
        f"{result['test_accuracy_change_vs_baseline_percentage_points']:+.2f} "
        f"percentage points"
    )

    print("\nSaved files:")
    print(f"- {RESULT_JSON}")
    print(f"- {TRIAL_CSV}")
    print(f"- {BEST_MODEL_DIR / 'final_model'}")


if __name__ == "__main__":
    main()
