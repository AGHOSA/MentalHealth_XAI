import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
# 1. LOAD PREPROCESSED DATASET
# ============================================================

data_path = "data/pirina_reddit_preprocessed.csv"

df = pd.read_csv(data_path)

print("========== DATASET ==========")
print("Shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 2. SELECT INPUT AND TARGET
# ============================================================

X = df["clean_text"]
y = df["label"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 4. TF-IDF FEATURE EXTRACTION
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

# IMPORTANT:
# Fit ONLY on training data to avoid data leakage.
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data using the already-fitted vectorizer.
X_test_tfidf = tfidf.transform(X_test)

print("\n========== TF-IDF ==========")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape :", X_test_tfidf.shape)


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train_tfidf, y_train)


# ============================================================
# 6. PREDICTION
# ============================================================

y_pred = lr.predict(X_test_tfidf)


# ============================================================
# 7. EVALUATION
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


print("\n========== LOGISTIC REGRESSION RESULTS ==========")

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

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 10. MOST IMPORTANT FEATURES
# ============================================================

feature_names = tfidf.get_feature_names_out()

coefficients = lr.coef_[0]

top_positive = coefficients.argsort()[-20:][::-1]
top_negative = coefficients.argsort()[:20]

print("\n========== TOP DEPRESSION-ASSOCIATED FEATURES ==========")

for index in top_positive:
    print(
        f"{feature_names[index]:30s} "
        f"{coefficients[index]:.4f}"
    )


print("\n========== TOP NON-DEPRESSION-ASSOCIATED FEATURES ==========")

for index in top_negative:
    print(
        f"{feature_names[index]:30s} "
        f"{coefficients[index]:.4f}"
    )