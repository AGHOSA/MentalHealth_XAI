import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
MODEL_NAME = "bert-base-uncased"
DATA_PATH = Path("data/pirina_reddit_preprocessed.csv")
OUTPUT_DIR = Path("results/models/bert_reddit")
REPORT_PATH = Path("results/bert_classification_report.csv")
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# -----------------------------
# Reproducibility and device
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# -----------------------------
# Load and validate dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df[["text", "label"]].dropna().copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df["label"] = df["label"].astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Expected binary labels: 0 and 1.")

print(f"Dataset size: {len(df)}")
print("Label counts:")
print(df["label"].value_counts().sort_index())

# 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

# Split remaining 20% into 10% validation and 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"],
)

print(
    f"Split sizes -> Train: {len(train_df)}, "
    f"Validation: {len(val_df)}, Test: {len(test_df)}"
)

# -----------------------------
# Tokenization
# -----------------------------
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
    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset

train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)
test_dataset = make_dataset(test_df)

# -----------------------------
# Model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "non_depression", 1: "depression"},
    label2id={"non_depression": 0, "depression": 1},
)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
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

# -----------------------------
# Training
# -----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=25,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------
# Final test evaluation
# -----------------------------
test_output = trainer.predict(test_dataset)
test_predictions = np.argmax(test_output.predictions, axis=-1)
test_labels = test_output.label_ids

accuracy = accuracy_score(test_labels, test_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0,
)

metrics = {
    "model": MODEL_NAME,
    "seed": SEED,
    "dataset_size": len(df),
    "train_size": len(train_df),
    "validation_size": len(val_df),
    "test_size": len(test_df),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "confusion_matrix": confusion_matrix(
        test_labels, test_predictions
    ).tolist(),
}

print("\nFinal test metrics:")
print(json.dumps(metrics, indent=2))

print("\nClassification report:")
print(
    classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["non_depression", "depression"],
        zero_division=0,
    )
)

report = pd.DataFrame(
    classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["non_depression", "depression"],
        output_dict=True,
        zero_division=0,
    )
).transpose()

report.to_csv(REPORT_PATH)
(OUTPUT_DIR / "test_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

trainer.save_model(str(OUTPUT_DIR / "final_model"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))

print(f"\nModel saved to: {OUTPUT_DIR / 'final_model'}")
print(f"Metrics saved to: {OUTPUT_DIR / 'test_metrics.json'}")
print(f"Classification report saved to: {REPORT_PATH}")
