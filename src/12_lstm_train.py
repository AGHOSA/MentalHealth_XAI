import os
import pickle
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Random seed:", SEED)


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

X_train_full = np.load(
    "results/lstm/X_train_padded.npy"
)

X_test = np.load(
    "results/lstm/X_test_padded.npy"
)

y_train_full = np.load(
    "results/lstm/y_train.npy"
)

y_test = np.load(
    "results/lstm/y_test.npy"
)

print("\n========== LSTM DATA ==========")
print("X_train_full:", X_train_full.shape)
print("X_test      :", X_test.shape)
print("y_train_full:", y_train_full.shape)
print("y_test      :", y_test.shape)


# ============================================================
# 3. CREATE VALIDATION SET
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.10,
    random_state=SEED,
    stratify=y_train_full
)

print("\n========== TRAIN / VALIDATION / TEST ==========")
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)

print("\nTraining class distribution:")
print(np.bincount(y_train))

print("\nValidation class distribution:")
print(np.bincount(y_val))

print("\nTesting class distribution:")
print(np.bincount(y_test))


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

with open(
    "results/lstm/tokenizer.pkl",
    "rb"
) as file:
    tokenizer = pickle.load(file)

MAX_WORDS = 10000
MAX_LEN = 200

print("\n========== TOKENIZER ==========")
print("Vocabulary size:", len(tokenizer.word_index))
print("MAX_WORDS:", MAX_WORDS)
print("MAX_LEN:", MAX_LEN)


# ============================================================
# 5. CALCULATE CLASS WEIGHTS
# ============================================================

class_0_count = np.sum(y_train == 0)
class_1_count = np.sum(y_train == 1)

total = len(y_train)

class_weight = {
    0: total / (2 * class_0_count),
    1: total / (2 * class_1_count)
}

print("\n========== CLASS WEIGHTS ==========")
print("Class 0 weight:", class_weight[0])
print("Class 1 weight:", class_weight[1])


# ============================================================
# 6. BUILD LSTM MODEL
# ============================================================

model = Sequential([
    Input(shape=(MAX_LEN,)),

    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        mask_zero=True
    ),

    LSTM(
        64
    ),

    Dropout(
        0.30
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. MODEL SUMMARY
# ============================================================

print("\n========== MODEL ARCHITECTURE ==========")

model.summary()


# ============================================================
# 9. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-5,
    verbose=1
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n========== TRAINING LSTM ==========")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# 11. VALIDATION PROBABILITIES
# ============================================================

print("\n========== VALIDATION THRESHOLD SEARCH ==========")

val_probabilities = model.predict(
    X_val,
    verbose=0
).flatten()


# ============================================================
# 12. FIND BEST THRESHOLD USING VALIDATION SET
# ============================================================

best_threshold = 0.50
best_val_accuracy = 0.0

for threshold in np.arange(
    0.30,
    0.71,
    0.01
):

    val_pred = (
        val_probabilities >= threshold
    ).astype(int)

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_threshold = threshold


print(
    f"Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f} "
    f"({best_val_accuracy * 100:.2f}%)"
)


# ============================================================
# 13. TEST PREDICTION
# ============================================================

print("\n========== TEST PREDICTION ==========")

test_probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

y_pred = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# 14. EVALUATION
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


print("\n========== FINAL LSTM RESULTS ==========")

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


# ============================================================
# 15. CLASSIFICATION REPORT
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
# 16. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

os.makedirs(
    "results/lstm",
    exist_ok=True
)

model_path = (
    "results/lstm/lstm_model.keras"
)

model.save(
    model_path
)

print("\n========== MODEL SAVED ==========")
print("Saved to:", model_path)


# ============================================================
# 18. SAVE THRESHOLD
# ============================================================

with open(
    "results/lstm/best_threshold.pkl",
    "wb"
) as file:

    pickle.dump(
        best_threshold,
        file
    )

print(
    "Threshold saved to:",
    "results/lstm/best_threshold.pkl"
)


# ============================================================
# 19. SAVE TRAINING HISTORY
# ============================================================

with open(
    "results/lstm/training_history.pkl",
    "wb"
) as file:

    pickle.dump(
        history.history,
        file
    )

print(
    "Training history saved to:",
    "results/lstm/training_history.pkl"
)


print("\n===================================================")
print("LSTM TRAINING COMPLETE")
print("===================================================")