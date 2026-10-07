import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

df = pd.read_csv(
    "data/pirina_reddit_preprocessed.csv"
)

X = df["clean_text"].astype(str)
y = df["label"].astype(int)


print("========== DATASET ==========")
print("Total samples:", len(df))
print("Class distribution:")
print(y.value_counts())


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

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 3. TEXT TOKENIZATION
# ============================================================

MAX_WORDS = 10000

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

# IMPORTANT:
# Fit tokenizer ONLY on training text.
tokenizer.fit_on_texts(X_train)


# ============================================================
# 4. CONVERT TEXT TO INTEGER SEQUENCES
# ============================================================

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)


print("\n========== TOKENIZATION ==========")
print("Vocabulary size:", len(tokenizer.word_index))
print("Training sequences:", len(X_train_sequences))
print("Testing sequences :", len(X_test_sequences))


# ============================================================
# 5. CHECK SEQUENCE LENGTHS
# ============================================================

train_lengths = [
    len(sequence)
    for sequence in X_train_sequences
]

print("\n========== SEQUENCE LENGTH ==========")
print("Minimum:", min(train_lengths))
print("Maximum:", max(train_lengths))
print("Mean   :", np.mean(train_lengths))


# ============================================================
# 6. PAD SEQUENCES
# ============================================================

MAX_LEN = 200

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)


print("\n========== PADDING ==========")
print("Maximum sequence length:", MAX_LEN)

print(
    "Training padded shape:",
    X_train_padded.shape
)

print(
    "Testing padded shape :",
    X_test_padded.shape
)


# ============================================================
# 7. CONVERT LABELS
# ============================================================

y_train_np = np.array(y_train)
y_test_np = np.array(y_test)


# ============================================================
# 8. CREATE SAVE DIRECTORY
# ============================================================

os.makedirs(
    "results/lstm",
    exist_ok=True
)


# ============================================================
# 9. SAVE TOKENIZER
# ============================================================

with open(
    "results/lstm/tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


# ============================================================
# 10. SAVE PREPARED ARRAYS
# ============================================================

np.save(
    "results/lstm/X_train_padded.npy",
    X_train_padded
)

np.save(
    "results/lstm/X_test_padded.npy",
    X_test_padded
)

np.save(
    "results/lstm/y_train.npy",
    y_train_np
)

np.save(
    "results/lstm/y_test.npy",
    y_test_np
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n========== LSTM PREPARATION COMPLETE ==========")

print(
    "Tokenizer saved to:",
    "results/lstm/tokenizer.pkl"
)

print(
    "Training data saved to:",
    "results/lstm/X_train_padded.npy"
)

print(
    "Testing data saved to:",
    "results/lstm/X_test_padded.npy"
)

print(
    "Labels saved to:",
    "results/lstm/y_train.npy and y_test.npy"
)