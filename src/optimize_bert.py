
import gc
import json
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
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
BASELINE_MODEL = Path("results/models/bert_reddit/final_model")
OUTPUT_DIR = Path("results/optimization/bert_optuna")

MAX_LENGTH = 128
N_TRIALS = 12
BATCH_SIZE = 8

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
# Validate paths
# ==================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not BASELINE_MODEL.exists():
    raise FileNotFoundError(
        f"Baseline model not found: {BASELINE_MODEL}"
    )

# ==================================================
# Load and validate dataset
# ==================================================
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

# Reproduce the original 80/10/10 split.
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
# Use the same tokenizer as the pretrained checkpoint.
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
# Optuna objective
# ==================================================
def objective(trial):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters to optimize.
    learning_rate = trial.suggest_float(
        "learning_rate",
        5e-6,
        5e-5,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        0.0,
        0.1,
    )

    warmup_steps = trial.suggest_int(
        "warmup_steps",
        0,
        200,
        step=20,
    )

    epochs = trial.suggest_int(
        "epochs",
        2,
        4,
    )

    trial_dir = (
        OUTPUT_DIR
        / "trials"
        / f"trial_{trial.number:03d}"
    )

    # Every trial starts from the same pretrained weights.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "non_depression", 1: "depression"},
        label2id={"non_depression": 0, "depression": 1},
    )

    args = TrainingArguments(
        output_dir=str(trial_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
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

        # load_best_model_at_end selects the best validation checkpoint.
        metrics = trainer.evaluate()
        score = float(metrics["eval_macro_f1"])

        trial.set_user_attr(
            "validation_accuracy",
            float(metrics["eval_accuracy"]),
        )
        trial.set_user_attr(
            "validation_macro_precision",
            float(metrics["eval_macro_precision"]),
        )
        trial.set_user_attr(
            "validation_macro_recall",
            float(metrics["eval_macro_recall"]),
        )
        trial.set_user_attr(
            "best_checkpoint",
            trainer.state.best_model_checkpoint,
        )

        print(
            f"\nTrial {trial.number}: "
            f"Validation Macro-F1 = {score:.4f}"
        )

        return score

    finally:
        del trainer
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================================================
# Run hyperparameter optimization
# ==================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

study = optuna.create_study(
    study_name="bert_depression_tpe_optimization",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    gc_after_trial=True,
)

# ==================================================
# Save results
# ==================================================
best_results = {
    "optimization_method": "Optuna TPE sampler",
    "objective": "validation_macro_f1",
    "pretrained_model": MODEL_NAME,
    "baseline_model_preserved": str(BASELINE_MODEL),
    "dataset_size": len(df),
    "train_size": len(train_df),
    "validation_size": len(val_df),
    "reserved_test_size": len(test_df),
    "number_of_trials": len(study.trials),
    "best_validation_macro_f1": float(study.best_value),
    "best_parameters": study.best_params,
}

(OUTPUT_DIR / "best_parameters.json").write_text(
    json.dumps(best_results, indent=2),
    encoding="utf-8",
)

trial_rows = []

for trial in study.trials:
    row = {
        "trial": trial.number,
        "state": trial.state.name,
        "validation_macro_f1": trial.value,
        **trial.params,
        **trial.user_attrs,
    }
    trial_rows.append(row)

pd.DataFrame(trial_rows).to_csv(
    OUTPUT_DIR / "optimization_trials.csv",
    index=False,
)

print("\nOptimization completed.")
print(json.dumps(best_results, indent=2))

print(f"\nResults saved in: {OUTPUT_DIR}")
print(
    "The reserved test set was not evaluated "
    "during hyperparameter optimization."
)
