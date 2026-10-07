import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    RFE,
    SelectFromModel
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. CREATE RESULT FOLDERS
# ============================================================

os.makedirs("results/models", exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("data/pirina_reddit_preprocessed.csv")

X = df["clean_text"]
y = df["label"]

print("========== DATASET ==========")
print("Shape:", df.shape)


# ============================================================
# 3. SAME TRAIN / TEST SPLIT
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


# ============================================================
# 4. TF-IDF
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
# 5. RESULTS LIST
# ============================================================

results = []


# ============================================================
# HELPER FUNCTION
# ============================================================

def evaluate_model(name, model, X_test_data, y_test_data):
    """
    Train, predict, evaluate and return metrics.
    """

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(
        y_test_data,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test_data,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test_data,
        y_pred,
        zero_division=0
    )

    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    return model


# ============================================================
# 6. BASELINE LOGISTIC REGRESSION
# ============================================================

print("\n========== 1. TF-IDF + LOGISTIC REGRESSION ==========")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train_tfidf, y_train)

y_pred = lr.predict(X_test_tfidf)

lr_metrics = {
    "Model": "TF-IDF + Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0)
}

results.append(lr_metrics)

print(f"Accuracy : {lr_metrics['Accuracy']:.4f}")
print(f"Precision: {lr_metrics['Precision']:.4f}")
print(f"Recall   : {lr_metrics['Recall']:.4f}")
print(f"F1-score : {lr_metrics['F1_Score']:.4f}")

joblib.dump(
    {
        "vectorizer": tfidf,
        "model": lr
    },
    "results/models/lr_baseline.joblib"
)

print("Saved: results/models/lr_baseline.joblib")


# ============================================================
# 7. CHI-SQUARE + LR
# ============================================================

print("\n========== 2. CHI-SQUARE + LOGISTIC REGRESSION ==========")

chi_selector = SelectKBest(
    score_func=chi2,
    k=2000
)

X_train_chi = chi_selector.fit_transform(
    X_train_tfidf,
    y_train
)

X_test_chi = chi_selector.transform(
    X_test_tfidf
)

chi_lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

chi_lr.fit(X_train_chi, y_train)

y_pred = chi_lr.predict(X_test_chi)

chi_metrics = {
    "Model": "Chi-Square + Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0)
}

results.append(chi_metrics)

print(f"Accuracy : {chi_metrics['Accuracy']:.4f}")
print(f"Precision: {chi_metrics['Precision']:.4f}")
print(f"Recall   : {chi_metrics['Recall']:.4f}")
print(f"F1-score : {chi_metrics['F1_Score']:.4f}")

joblib.dump(
    {
        "vectorizer": tfidf,
        "selector": chi_selector,
        "model": chi_lr
    },
    "results/models/chi2_lr.joblib"
)

print("Saved: results/models/chi2_lr.joblib")


# ============================================================
# 8. RFE + LR
# ============================================================

print("\n========== 3. RFE + LOGISTIC REGRESSION ==========")
print("Starting RFE...")

rfe_estimator = LogisticRegression(
    max_iter=1000,
    random_state=42
)

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=2000,
    step=0.1
)

X_train_rfe = rfe.fit_transform(
    X_train_tfidf,
    y_train
)

X_test_rfe = rfe.transform(
    X_test_tfidf
)

rfe_lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

rfe_lr.fit(X_train_rfe, y_train)

y_pred = rfe_lr.predict(X_test_rfe)

rfe_metrics = {
    "Model": "RFE + Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0)
}

results.append(rfe_metrics)

print(f"Accuracy : {rfe_metrics['Accuracy']:.4f}")
print(f"Precision: {rfe_metrics['Precision']:.4f}")
print(f"Recall   : {rfe_metrics['Recall']:.4f}")
print(f"F1-score : {rfe_metrics['F1_Score']:.4f}")

joblib.dump(
    {
        "vectorizer": tfidf,
        "selector": rfe,
        "model": rfe_lr
    },
    "results/models/rfe_lr.joblib"
)

print("Saved: results/models/rfe_lr.joblib")


# ============================================================
# 9. L1 + LR
# ============================================================

print("\n========== 4. L1 + LOGISTIC REGRESSION ==========")

l1_selector_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=1000,
    random_state=42
)

l1_selector = SelectFromModel(
    estimator=l1_selector_model,
    prefit=False
)

X_train_l1 = l1_selector.fit_transform(
    X_train_tfidf,
    y_train
)

X_test_l1 = l1_selector.transform(
    X_test_tfidf
)

print("Selected features:", X_train_l1.shape[1])

l1_lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

l1_lr.fit(X_train_l1, y_train)

y_pred = l1_lr.predict(X_test_l1)

l1_metrics = {
    "Model": "L1 + Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0)
}

results.append(l1_metrics)

print(f"Accuracy : {l1_metrics['Accuracy']:.4f}")
print(f"Precision: {l1_metrics['Precision']:.4f}")
print(f"Recall   : {l1_metrics['Recall']:.4f}")
print(f"F1-score : {l1_metrics['F1_Score']:.4f}")

joblib.dump(
    {
        "vectorizer": tfidf,
        "selector": l1_selector,
        "model": l1_lr
    },
    "results/models/l1_lr.joblib"
)

print("Saved: results/models/l1_lr.joblib")


# ============================================================
# 10. NAIVE BAYES
# ============================================================

print("\n========== 5. TF-IDF + NAIVE BAYES ==========")

nb = MultinomialNB()

nb.fit(
    X_train_tfidf,
    y_train
)

y_pred = nb.predict(X_test_tfidf)

nb_metrics = {
    "Model": "TF-IDF + Naive Bayes",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0)
}

results.append(nb_metrics)

print(f"Accuracy : {nb_metrics['Accuracy']:.4f}")
print(f"Precision: {nb_metrics['Precision']:.4f}")
print(f"Recall   : {nb_metrics['Recall']:.4f}")
print(f"F1-score : {nb_metrics['F1_Score']:.4f}")

joblib.dump(
    {
        "vectorizer": tfidf,
        "model": nb
    },
    "results/models/naive_bayes.joblib"
)

print("Saved: results/models/naive_bayes.joblib")


# ============================================================
# 11. SVM
# ============================================================

print("\n========== 6. TF-IDF + SVM ==========")

svm = LinearSVC(
    random_state=42
)

svm.fit(
    X_train_tfidf,
    y_train
)

y_pred = svm.predict(X_test_tfidf)

svm_metrics = {
    "Model": "TF-IDF + SVM",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0)
}

results.append(svm_metrics)

print(f"Accuracy : {svm_metrics['Accuracy']:.4f}")
print(f"Precision: {svm_metrics['Precision']:.4f}")
print(f"Recall   : {svm_metrics['Recall']:.4f}")
print(f"F1-score : {svm_metrics['F1_Score']:.4f}")

joblib.dump(
    {
        "vectorizer": tfidf,
        "model": svm
    },
    "results/models/svm.joblib"
)

print("Saved: results/models/svm.joblib")


# ============================================================
# 12. RANDOM FOREST
# ============================================================

print("\n========== 7. TF-IDF + RANDOM FOREST ==========")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_tfidf,
    y_train
)

y_pred = rf.predict(X_test_tfidf)

rf_metrics = {
    "Model": "TF-IDF + Random Forest",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1_Score": f1_score(y_test, y_pred, zero_division=0)
}

results.append(rf_metrics)

print(f"Accuracy : {rf_metrics['Accuracy']:.4f}")
print(f"Precision: {rf_metrics['Precision']:.4f}")
print(f"Recall   : {rf_metrics['Recall']:.4f}")
print(f"F1-score : {rf_metrics['F1_Score']:.4f}")

joblib.dump(
    {
        "vectorizer": tfidf,
        "model": rf
    },
    "results/models/random_forest.joblib"
)

print("Saved: results/models/random_forest.joblib")


# ============================================================
# 13. SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/ml_results.csv",
    index=False
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n\n============================================================")
print("ALL CLASSICAL ML MODELS TRAINED AND SAVED")
print("============================================================")

print("\nResults:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nModels saved in:")
print("results/models/")

print("\nResults saved in:")
print("results/ml_results.csv")

print("\n============================================================")