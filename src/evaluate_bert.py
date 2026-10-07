
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "pirina_reddit_preprocessed.csv"
MODEL_PATH = PROJECT_DIR / "results" / "models" / "bert_reddit" / "final_model"
OUTPUT_DIR = PROJECT_DIR / "results" / "bert_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16

LABEL_NAMES = {
    0: "non_depression",
    1: "depression",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# --------------------------------------------------
# 2. LOAD AND CLEAN DATA
# Keep preprocessing consistent with training
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df = df[["text", "label"]].dropna().copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()
df["label"] = df["label"].astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Expected binary labels 0 and 1.")

# Reproduce the original 80/10/10 stratified split.
# This assumes the dataset and row order are unchanged
# since the original training run.
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

test_df = test_df.reset_index(drop=True)

print(f"Full dataset: {len(df)}")
print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")
print(f"Test rows: {len(test_df)}")


# --------------------------------------------------
# 3. LOAD SAVED MODEL AND TOKENIZER
# --------------------------------------------------

print("\nLoading saved BERT model...")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH)
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# 4. TOKENIZE TEST DATA AND PREDICT
# --------------------------------------------------

texts = test_df["text"].tolist()
y_true = test_df["label"].to_numpy()

all_probabilities = []

with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=1)

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

probabilities = np.concatenate(all_probabilities, axis=0)
y_pred = np.argmax(probabilities, axis=1)
p_depression = probabilities[:, 1]


# --------------------------------------------------
# 5. CALCULATE METRICS
# --------------------------------------------------

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

metrics = {
    "model": "bert-base-uncased fine-tuned on Reddit dataset",
    "device": str(device),
    "seed": SEED,
    "test_samples": int(len(y_true)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_true, y_pred)
    ),
    "macro_precision": float(
        precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    ),
    "macro_recall": float(
        recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    ),
    "macro_f1": float(
        f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    ),
    "confusion_matrix_labels": [
        "non_depression",
        "depression",
    ],
    "confusion_matrix": cm.tolist(),
}

# ROC-AUC requires both classes to exist in the test set.
if len(np.unique(y_true)) == 2:
    metrics["roc_auc"] = float(
        roc_auc_score(y_true, p_depression)
    )
else:
    metrics["roc_auc"] = None

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["non_depression", "depression"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv",
    index=True,
)

with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# 6. SAVE CONFUSION MATRIX IMAGE
# --------------------------------------------------

row_totals = cm.sum(axis=1, keepdims=True)
row_percentages = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0,
) * 100

annotations = np.empty_like(cm, dtype=object)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annotations[i, j] = (
            f"{cm[i, j]}\n"
            f"({row_percentages[i, j]:.1f}%)"
        )

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=annotations,
    fmt="",
    cmap="Blues",
    xticklabels=["Non-depression", "Depression"],
    yticklabels=["Non-depression", "Depression"],
)
plt.title("BERT Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# --------------------------------------------------
# 7. SAVE ROC CURVE
# --------------------------------------------------

if len(np.unique(y_true)) == 2:
    fpr, tpr, _ = roc_curve(y_true, p_depression)
    auc_value = metrics["roc_auc"]

    plt.figure(figsize=(7, 5))
    plt.plot(
        fpr,
        tpr,
        label=f"BERT ROC-AUC = {auc_value:.4f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("BERT ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# --------------------------------------------------
# 8. SAVE INDIVIDUAL TEST PREDICTIONS
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "text": texts,
    "actual_label": [
        LABEL_NAMES[int(label)] for label in y_true
    ],
    "predicted_label": [
        LABEL_NAMES[int(label)] for label in y_pred
    ],
    "probability_non_depression": probabilities[:, 0],
    "probability_depression": probabilities[:, 1],
    "correct_prediction": (y_true == y_pred),
})

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)


# --------------------------------------------------
# 9. PRINT FINAL RESULTS
# --------------------------------------------------

print("\n========== BERT EVALUATION RESULTS ==========")
print(f"Test samples:       {metrics['test_samples']}")
print(f"Accuracy:           {metrics['accuracy']:.4f}")
print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")
print(f"Macro precision:    {metrics['macro_precision']:.4f}")
print(f"Macro recall:       {metrics['macro_recall']:.4f}")
print(f"Macro F1-score:     {metrics['macro_f1']:.4f}")

if metrics["roc_auc"] is not None:
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(f"\nAll evaluation files saved to:\n{OUTPUT_DIR}")
print("\nEvaluation completed.")
