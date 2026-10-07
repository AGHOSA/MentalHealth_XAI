import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD SAVED LSTM MODEL
# ============================================================

model = load_model(
    "results/lstm/lstm_model.keras"
)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

X_test = np.load(
    "results/lstm/X_test_padded.npy"
)

y_test = np.load(
    "results/lstm/y_test.npy"
)


# ============================================================
# 3. LOAD SAVED THRESHOLD
# ============================================================

with open(
    "results/lstm/best_threshold.pkl",
    "rb"
) as file:

    threshold = pickle.load(file)


# ============================================================
# 4. PREDICTION
# ============================================================

probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

y_pred = (
    probabilities >= threshold
).astype(int)


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 6. CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame([
    {
        "Model": "LSTM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Threshold": threshold
    }
])


# ============================================================
# 7. SAVE CSV
# ============================================================

output_path = "results/lstm/lstm_results.csv"

results.to_csv(
    output_path,
    index=False
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========== VERIFIED LSTM RESULTS ==========")

print(
    f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f} ({precision * 100:.2f}%)"
)

print(
    f"Recall   : {recall:.4f} ({recall * 100:.2f}%)"
)

print(
    f"F1-score : {f1:.4f} ({f1 * 100:.2f}%)"
)

print(
    f"Threshold: {threshold:.2f}"
)

print("\nConfusion Matrix:")
print(cm)

print(
    "\nSaved to:",
    output_path
)