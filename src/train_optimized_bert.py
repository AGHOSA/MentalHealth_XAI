
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
OPTUNA_RESULTS = Path(
    "results/optimization/bert_optuna/best_parameters.json"
)
OUTPUT_DIR = Path("results/models/bert_reddit_optuna")
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
# Validate input files
# ==================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not OPTUNA_RESULTS.exists():
    raise FileNotFoundError(
        f"Optuna results not found: {OPTUNA_RESULTS}"
    )

if FINAL_MODEL_DIR.exists():
    raise FileExistsError(
        f"Output model already exists: {FINAL_MODEL_DIR}\n"
        "Rename or back up the existing folder before rerunning."
    )

with OPTUNA_RESULTS.open("r", encoding="utf-8") as file:
    optuna_results = json.load(file)

best_params = optuna_results["best_parameters"]

learning_rate = float(best_params["learning_rate"])
weight_decay = float(best_params["weight_decay"])
warmup_steps = int(best_params["warmup_steps"])
epochs = int(best_params["epochs"])

print("\nSelected hyperparameters:")
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

# Reproduce the same split used during optimization.
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
    f"\nTrain: {len(train_df)}"
    f"\nValidation: {len(val_df)}"
    f"\nReserved test: {len(test_df)}"
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
    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["text"])
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
# Train final model with best hyperparameters
# ==================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=False)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "non_depression", 1: "depression"},
    label2id={"non_depression": 0, "depression": 1},
)

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
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
)

print("\nStarting final model training...")
trainer.train()

# Save the checkpoint with the best validation Macro-F1.
trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

print(f"\nModel saved to: {FINAL_MODEL_DIR}")

# ==================================================
# Final evaluation on reserved test set
# ==================================================
print("\nEvaluating final model on reserved test data...")

test_output = trainer.predict(test_dataset)
test_logits = test_output.predictions
test_labels = test_output.label_ids
test_predictions = np.argmax(test_logits, axis=-1)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0,
)

test_metrics = {
    "accuracy": float(
        accuracy_score(test_labels, test_predictions)
    ),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "test_samples": int(len(test_labels)),
    "confusion_matrix": confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1],
    ).tolist(),
    "classification_report": classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=["non_depression", "depression"],
        output_dict=True,
        zero_division=0,
    ),
}

metrics_path = OUTPUT_DIR / "test_metrics.json"
metrics_path.write_text(
    json.dumps(test_metrics, indent=2),
    encoding="utf-8",
)

print("\nFinal test metrics:")
print(json.dumps(test_metrics, indent=2))
print(f"\nMetrics saved to: {metrics_path}")

# ==================================================
# Cleanup
# ==================================================
del trainer, model
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nFinal optimized BERT training and evaluation completed.")
