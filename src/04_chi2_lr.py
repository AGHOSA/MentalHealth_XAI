import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
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
# 1. LOAD PREPROCESSED DATA
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
print("Before feature selection:")
print("Train:", X_train_tfidf.shape)
print("Test :", X_test_tfidf.shape)


# ============================================================
# 4. CHI-SQUARE FEATURE SELECTION
# ============================================================

selector = SelectKBest(
    score_func=chi2,
    k=2000
)

X_train_selected = selector.fit_transform(
    X_train_tfidf,
    y_train
)

X_test_selected = selector.transform(
    X_test_tfidf
)

print("\n========== CHI-SQUARE ==========")
print("Selected features: 2000")
print("Train:", X_train_selected.shape)
print("Test :", X_test_selected.shape)


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(
    X_train_selected,
    y_train
)


# ============================================================
# 6. PREDICTION
# ============================================================

y_pred = lr.predict(X_test_selected)


# ============================================================
# 7. METRICS
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


print("\n========== CHI-SQUARE + LR RESULTS ==========")

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# 8. CLASSIFICATION REPORT
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
# 9. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)