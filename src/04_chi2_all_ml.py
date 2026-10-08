import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

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

print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

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

print("\n" + "=" * 70)
print("TF-IDF")
print("=" * 70)

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

print("\n" + "=" * 70)
print("CHI-SQUARE FEATURE SELECTION")
print("=" * 70)

print("Selected features: 2000")
print("Train:", X_train_selected.shape)
print("Test :", X_test_selected.shape)


# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression + Chi-Square":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "SVM + Chi-Square":
        LinearSVC(
            random_state=42
        ),

    "Naive Bayes + Chi-Square":
        MultinomialNB(),

    "Random Forest + Chi-Square":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
}


# ============================================================
# 6. TRAIN + EVALUATE
# ============================================================

results = []

os.makedirs("results/chi2_ml", exist_ok=True)


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Training...")

    model.fit(
        X_train_selected,
        y_train
    )

    print("Training completed.")

    # Prediction
    y_pred = model.predict(
        X_test_selected
    )

    # Metrics
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

    # Print results
    print("\nResults:")

    print(
        f"Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    # Classification report
    print("\nClassification Report:")

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

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Non-Depression", "Actual Depression"],
        columns=["Predicted Non-Depression", "Predicted Depression"]
    )

    safe_name = (
        model_name
        .lower()
        .replace(" + chi-square", "")
        .replace(" ", "_")
    )

    cm_df.to_csv(
        f"results/chi2_ml/{safe_name}_confusion_matrix.csv"
    )

    # Store result
    results.append({
        "Model": model_name,
        "Feature_Selection": "Chi-Square",
        "Selected_Features": 2000,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })


# ============================================================
# 7. SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df.to_csv(
    "results/chi2_ml/chi2_ml_comparison.csv",
    index=False
)


# ============================================================
# 8. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHI-SQUARE + ML COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(
    "results/chi2_ml/chi2_ml_comparison.csv"
)

print("\nExperiment completed successfully.")