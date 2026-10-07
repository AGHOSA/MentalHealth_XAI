import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/pirina_reddit_preprocessed.csv")

X = df["clean_text"]
y = df["label"]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("========== TRAIN / TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 3. TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\n========== TF-IDF ==========")
print("Train:", X_train_tfidf.shape)
print("Test :", X_test_tfidf.shape)


# ============================================================
# 4. BASE ESTIMATOR FOR RFE
# ============================================================

estimator = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 5. RFE FEATURE SELECTION
# ============================================================

rfe = RFE(
    estimator=estimator,
    n_features_to_select=2000,
    step=0.1
)

print("\n========== RFE ==========")
print("Starting RFE...")
print("Initial features:", X_train_tfidf.shape[1])
print("Target features :", 2000)

X_train_rfe = rfe.fit_transform(
    X_train_tfidf,
    y_train
)

X_test_rfe = rfe.transform(
    X_test_tfidf
)

print("RFE completed.")
print("Selected features:", X_train_rfe.shape[1])
print("Train:", X_train_rfe.shape)
print("Test :", X_test_rfe.shape)


# ============================================================
# 6. FINAL LOGISTIC REGRESSION
# ============================================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(
    X_train_rfe,
    y_train
)


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = lr.predict(X_test_rfe)


# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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


print("\n========== RFE + LR RESULTS ==========")

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Depression",
            "Depression"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)