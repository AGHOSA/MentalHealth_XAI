import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "pirina_reddit_preprocessed.csv"

MODELS = {
    "baseline": PROJECT_DIR / "results/models/bert_reddit/final_model",
    "optimized": PROJECT_DIR / "results/models/bert_reddit_optuna/final_model",
}

OUTPUT_DIR = PROJECT_DIR / "results/model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproduce the original stratified 80/10/10 split.
df = pd.read_csv(DATA_PATH)
df = df[["text", "label"]].dropna().copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()
df["label"] = df["label"].astype(int)

_, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"],
)

_, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"],
)

test_df = test_df.reset_index(drop=True)
texts = test_df["text"].tolist()
y_true = test_df["label"].to_numpy()

print(f"Device: {device}")
print(f"Test samples: {len(test_df)}")

all_predictions = {}
all_metrics = []

for model_name, model_path in MODELS.items():
    print(f"\nEvaluating {model_name}: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    ).to(device)
    model.eval()

    batch_predictions = []
    batch_probabilities = []

    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            batch_probabilities.append(probs)
            batch_predictions.extend(np.argmax(probs, axis=1))

    y_pred = np.array(batch_predictions)
    probabilities = np.concatenate(batch_probabilities, axis=0)

    all_predictions[model_name] = y_pred

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }
    all_metrics.append(metrics)

    pred_df = pd.DataFrame({
        "text": texts,
        "actual_label": y_true,
        "predicted_label": y_pred,
        "probability_non_depression": probabilities[:, 0],
        "probability_depression": probabilities[:, 1],
        "correct_prediction": y_true == y_pred,
    })
    pred_df.to_csv(
        OUTPUT_DIR / f"{model_name}_predictions.csv",
        index=False,
    )

    print(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Compare predictions on identical test rows.
comparison_df = pd.DataFrame({
    "text": texts,
    "actual_label": y_true,
    "baseline_prediction": all_predictions["baseline"],
    "optimized_prediction": all_predictions["optimized"],
})

comparison_df["baseline_correct"] = (
    comparison_df["actual_label"] == comparison_df["baseline_prediction"]
)
comparison_df["optimized_correct"] = (
    comparison_df["actual_label"] == comparison_df["optimized_prediction"]
)

comparison_df["prediction_changed"] = (
    comparison_df["baseline_prediction"]
    != comparison_df["optimized_prediction"]
)

comparison_df["baseline_only_correct"] = (
    comparison_df["baseline_correct"]
    & ~comparison_df["optimized_correct"]
)

comparison_df["optimized_only_correct"] = (
    ~comparison_df["baseline_correct"]
    & comparison_df["optimized_correct"]
)

comparison_df.to_csv(
    OUTPUT_DIR / "prediction_comparison.csv",
    index=False,
)

with open(OUTPUT_DIR / "metrics_comparison.json", "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)

print("\n========== MODEL COMPARISON ==========")
print(pd.DataFrame(all_metrics)[
    ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
].to_string(index=False))

print("\nPredictions changed:", int(comparison_df["prediction_changed"].sum()))
print("Baseline only correct:", int(comparison_df["baseline_only_correct"].sum()))
print("Optimized only correct:", int(comparison_df["optimized_only_correct"].sum()))

print(f"\nResults saved to: {OUTPUT_DIR}")