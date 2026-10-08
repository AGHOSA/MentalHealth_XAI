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
    Trainer,
    TrainingArguments,
)


# ==================================================
# Configuration
# ==================================================
SEED = 42
MODEL_NAME = "bert-base-uncased"

DATA_PATH = Path(
    "data/pirina_reddit_preprocessed.csv"
)

BASELINE_MODEL_DIR = Path(
    "results/models/bert_reddit/final_model"
)

OUTPUT_DIR = Path(
    "results/optimization/bert_threshold"
)

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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

if device == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ==================================================
# Validate files
# ==================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not BASELINE_MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Baseline BERT model not found: "
        f"{BASELINE_MODEL_DIR}"
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: "
        f"{OUTPUT_DIR}\n"
        "Rename or remove it before rerunning."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)


# ==================================================
# Load dataset
# ==================================================
df = pd.read_csv(DATA_PATH)

df = df[
    ["text", "label"]
].dropna().copy()

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)

df = df[
    df["text"] != ""
].copy()

df["label"] = df[
    "label"
].astype(int)

if set(df["label"].unique()) != {0, 1}:
    raise ValueError(
        "Expected binary labels 0 and 1."
    )

print(
    f"\nTotal dataset samples: {len(df)}"
)


# ==================================================
# EXACT SAME SPLIT USED FOR BERT EXPERIMENTS
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
    f"\nTrain samples:      {len(train_df)}"
    f"\nValidation samples: {len(val_df)}"
    f"\nReserved test:      {len(test_df)}"
)

print(
    "\nIMPORTANT: "
    "The reserved test set will only be used "
    "after selecting the threshold."
)


# ==================================================
# Tokenizer
# ==================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASELINE_MODEL_DIR
)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def make_dataset(frame):
    dataset = Dataset.from_pandas(
        frame[
            ["text", "label"]
        ].reset_index(drop=True)
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


train_dataset = make_dataset(
    train_df
)

val_dataset = make_dataset(
    val_df
)

test_dataset = make_dataset(
    test_df
)


# ==================================================
# Load EXISTING baseline BERT
# ==================================================
print(
    "\nLoading existing baseline BERT..."
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASELINE_MODEL_DIR
    )
)

model.to(device)


# ==================================================
# Trainer for inference only
# ==================================================
args = TrainingArguments(
    output_dir=str(
        OUTPUT_DIR / "inference"
    ),
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
)


# ==================================================
# Get validation probabilities
# ==================================================
print(
    "\nGenerating validation predictions..."
)

val_output = trainer.predict(
    val_dataset
)

val_logits = val_output.predictions
val_labels = val_output.label_ids

val_probabilities = torch.softmax(
    torch.tensor(val_logits),
    dim=1,
).numpy()[:, 1]


# ==================================================
# Threshold search
# ==================================================
print(
    "\nSearching for optimal decision threshold..."
)

thresholds = np.arange(
    0.30,
    0.701,
    0.01,
)

threshold_results = []

best_threshold = 0.50
best_macro_f1 = -1.0
best_accuracy = 0.0

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        val_labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            val_labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    threshold_results.append(
        {
            "threshold": float(
                threshold
            ),
            "accuracy": float(
                accuracy
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
        }
    )

    # Primary objective = validation Macro-F1
    if f1 > best_macro_f1:
        best_macro_f1 = f1
        best_accuracy = accuracy
        best_threshold = float(
            threshold
        )


# ==================================================
# Save threshold search results
# ==================================================
threshold_results_path = (
    OUTPUT_DIR /
    "threshold_search_results.csv"
)

pd.DataFrame(
    threshold_results
).to_csv(
    threshold_results_path,
    index=False,
)


print(
    "\n" + "=" * 60
)

print(
    "BEST VALIDATION THRESHOLD"
)

print(
    "=" * 60
)

print(
    f"Threshold       : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation Acc. : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Validation F1   : "
    f"{best_macro_f1 * 100:.2f}%"
)


# ==================================================
# IMPORTANT:
# Threshold is now FROZEN.
# Test set is evaluated only once with this threshold.
# ==================================================
print(
    "\nThreshold selected from validation set."
)

print(
    "Now evaluating the untouched test set..."
)


# ==================================================
# Test probabilities
# ==================================================
test_output = trainer.predict(
    test_dataset
)

test_logits = test_output.predictions
test_labels = test_output.label_ids

test_probabilities = torch.softmax(
    torch.tensor(test_logits),
    dim=1,
).numpy()[:, 1]


# ==================================================
# Apply FROZEN threshold to test set
# ==================================================
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ==================================================
# Final test metrics
# ==================================================
test_accuracy = accuracy_score(
    test_labels,
    test_predictions,
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="macro",
        zero_division=0,
    )
)

test_confusion_matrix = (
    confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1],
    )
)

test_classification_report = (
    classification_report(
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
)


# ==================================================
# Save final results
# ==================================================
final_results = {
    "method": (
        "Validation-Based "
        "Decision Threshold Optimization"
    ),
    "base_model": "Baseline BERT",
    "pretrained_model": MODEL_NAME,

    "selected_threshold": (
        best_threshold
    ),

    "validation_accuracy": (
        float(best_accuracy)
    ),

    "validation_macro_f1": (
        float(best_macro_f1)
    ),

    "test_samples": int(
        len(test_labels)
    ),

    "test_accuracy": float(
        test_accuracy
    ),

    "test_macro_precision": float(
        precision
    ),

    "test_macro_recall": float(
        recall
    ),

    "test_macro_f1": float(
        f1
    ),

    "confusion_matrix": (
        test_confusion_matrix.tolist()
    ),

    "classification_report": (
        test_classification_report
    ),

    "test_set_used_for_threshold_selection": False,
}


final_results_path = (
    OUTPUT_DIR /
    "final_results.json"
)

final_results_path.write_text(
    json.dumps(
        final_results,
        indent=2,
    ),
    encoding="utf-8",
)


# ==================================================
# Save predictions
# ==================================================
predictions_df = pd.DataFrame(
    {
        "text": test_df[
            "text"
        ].values,
        "actual_label": test_labels,
        "depression_probability": (
            test_probabilities
        ),
        "predicted_label": (
            test_predictions
        ),
    }
)

predictions_df.to_csv(
    OUTPUT_DIR /
    "test_predictions.csv",
    index=False,
)


# ==================================================
# Print final results
# ==================================================
print(
    "\n" + "=" * 60
)

print(
    "THRESHOLD-OPTIMIZED BERT "
    "FINAL TEST RESULTS"
)

print(
    "=" * 60
)

print(
    f"\nSelected Threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Test Accuracy      : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision    : "
    f"{precision * 100:.2f}%"
)

print(
    f"Macro Recall       : "
    f"{recall * 100:.2f}%"
)

print(
    f"Macro F1           : "
    f"{f1 * 100:.2f}%"
)

print(
    "\nConfusion Matrix:"
)

print(
    test_confusion_matrix
)

print(
    f"\nResults saved to:"
    f"\n{final_results_path}"
)

print(
    f"\nThreshold search saved to:"
    f"\n{threshold_results_path}"
)


# ==================================================
# Cleanup
# ==================================================
del trainer
del model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


print(
    "\nThreshold optimization completed."
)