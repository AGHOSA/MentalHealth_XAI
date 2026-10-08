import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ==================================================
# Configuration
# ==================================================
SEED = 42
MODEL_NAME = "bert-base-uncased"

DATA_PATH = Path("data/pirina_reddit_preprocessed.csv")

# Best hyperparameters obtained from Genetic Algorithm
GA_RESULTS = {
    "learning_rate": 1.718692159596868e-05,
    "weight_decay": 0.09731157639793707,
    "warmup_ratio": 0.056780156581253016,
    "warmup_steps": 42,
    "epochs": 4,
}

# IMPORTANT:
# Use a new output directory so existing Optuna/Baseline results
# remain completely untouched.
OUTPUT_DIR = Path("results/models/bert_reddit_ga")
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"

MAX_LENGTH = 128
BATCH_SIZE = 8

# ==================================================
# Reproducibility
# ==================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================================================
# Validate input dataset
# ==================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if FINAL_MODEL_DIR.exists():
    raise FileExistsError(
        f"Output model already exists: {FINAL_MODEL_DIR}\n"
        "Rename or back up the existing folder before rerunning."
    )

# ==================================================
# Load GA-selected hyperparameters
# ==================================================
best_params = GA_RESULTS

learning_rate = float(best_params["learning_rate"])
weight_decay = float(best_params["weight_decay"])
warmup_steps = int(best_params["warmup_steps"])
epochs = int(best_params["epochs"])

print("\n" + "=" * 60)
print("GA-OPTIMIZED BERT FINAL TRAINING")
print("=" * 60)

print("\nSelected GA hyperparameters:")
print(json.dumps(best_params, indent=2))

# ==================================================
# Load and validate dataset
# ==================================================
df = pd.read_csv(DATA_PATH)

df = df[["text", "label"]].dropna().copy()

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()

df["label"] = df["label"].astype(int)

if set(df["label"].unique()) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

print(f"\nTotal dataset samples: {len(df)}")

# ==================================================
# Reproduce EXACT same split used during GA
# ==================================================
# 80% training
# 20% temporary set
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

# Temporary set is split equally:
# 10% validation
# 10% reserved test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"],
)

print(
    f"\nTrain samples:     {len(train_df)}"
    f"\nValidation samples:{len(val_df)}"
    f"\nReserved test:     {len(test_df)}"
)

print("\nIMPORTANT:")
print("The reserved test set was NOT used during GA optimization.")

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

    dataset = dataset.rename_column(
        "label",
        "labels",
    )

    dataset = dataset.remove_columns(
        ["text"]
    )

    dataset.set_format("torch")

    return dataset


train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)
test_dataset = make_dataset(test_df)

# ==================================================
# Metrics
# ==================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

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
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


# ==================================================
# Create output directories
# ==================================================
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

# ==================================================
# Load pretrained BERT
# ==================================================
print("\nLoading pretrained BERT model...")

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

# ==================================================
# Training Arguments
# ==================================================
args = TrainingArguments(
    output_dir=str(
        OUTPUT_DIR / "checkpoints"
    ),

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

# ==================================================
# Trainer
# ==================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ==================================================
# Final Training
# ==================================================
print("\n" + "=" * 60)
print("STARTING FINAL GA-OPTIMIZED BERT TRAINING")
print("=" * 60)

trainer.train()

# ==================================================
# Save Best Validation Checkpoint
# ==================================================
trainer.save_model(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)

print(
    f"\nBest GA-BERT model saved to:\n"
    f"{FINAL_MODEL_DIR}"
)

# ==================================================
# Final Evaluation on RESERVED TEST SET
# ==================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION ON RESERVED TEST SET")
print("=" * 60)

test_output = trainer.predict(
    test_dataset
)

test_logits = test_output.predictions
test_labels = test_output.label_ids

test_predictions = np.argmax(
    test_logits,
    axis=-1,
)

# ==================================================
# Test Metrics
# ==================================================
precision, recall, f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="macro",
        zero_division=0,
    )
)

test_accuracy = accuracy_score(
    test_labels,
    test_predictions,
)

test_confusion_matrix = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1],
)

test_classification_report = classification_report(
    test_labels,
    test_predictions,
    labels=[0, 1],
    target_names=[
        "non_depression",
        "depression",
    ],
    output_dict=True,
    zero_division=0,
)

test_metrics = {
    "model": "GA-Optimized BERT",
    "pretrained_model": MODEL_NAME,

    "optimization_method": "Genetic Algorithm",

    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "warmup_steps": warmup_steps,
    "epochs": epochs,

    "test_samples": int(
        len(test_labels)
    ),

    "accuracy": float(
        test_accuracy
    ),

    "macro_precision": float(
        precision
    ),

    "macro_recall": float(
        recall
    ),

    "macro_f1": float(
        f1
    ),

    "confusion_matrix": (
        test_confusion_matrix.tolist()
    ),

    "classification_report": (
        test_classification_report
    ),
}

# ==================================================
# Save Test Metrics
# ==================================================
metrics_path = (
    OUTPUT_DIR /
    "test_metrics.json"
)

metrics_path.write_text(
    json.dumps(
        test_metrics,
        indent=2,
    ),
    encoding="utf-8",
)

# ==================================================
# Print Final Results
# ==================================================
print("\n" + "=" * 60)
print("FINAL GA-BERT TEST RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Macro Recall   : "
    f"{recall * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{f1 * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(test_confusion_matrix)

print(
    f"\nMetrics saved to:\n"
    f"{metrics_path}"
)

# ==================================================
# Cleanup
# ==================================================
del trainer
del model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("FINAL GA-OPTIMIZED BERT TRAINING COMPLETED")
print("=" * 60)