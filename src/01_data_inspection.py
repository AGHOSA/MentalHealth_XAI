import pandas as pd

# Load dataset
df = pd.read_csv("data/pirina_reddit_public_corpus.csv")

print("========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== CLASS DISTRIBUTION ==========")
print(df["label"].value_counts())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE TEXTS ==========")
print(df["text"].duplicated().sum())

print("\n========== SAMPLE POSTS ==========")
for i in range(min(3, len(df))):
    print(f"\n--- Sample {i + 1} ---")
    print(df.loc[i, "text"][:500])
    print("Label:", df.loc[i, "label"])