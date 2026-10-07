import re
import pandas as pd
import nltk

from ftfy import fix_text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


# ============================================================
# NLTK RESOURCES
# ============================================================

nltk.download("stopwords")
nltk.download("wordnet")


# ============================================================
# 1. LOAD DATASET
# ============================================================

input_path = "data/pirina_reddit_public_corpus.csv"

df = pd.read_csv(input_path)

print("========== ORIGINAL DATASET ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. REMOVE DUPLICATE TEXTS
# ============================================================

before = len(df)

df = df.drop_duplicates(subset="text").reset_index(drop=True)

after = len(df)

print("\n========== DUPLICATE REMOVAL ==========")
print("Duplicates removed:", before - after)
print("Shape after duplicate removal:", df.shape)


# ============================================================
# 3. FIX TEXT ENCODING
# ============================================================

df["clean_text"] = (
    df["text"]
    .astype(str)
    .apply(fix_text)
)


# ============================================================
# 4. CONVERT TO LOWERCASE
# ============================================================

df["clean_text"] = df["clean_text"].str.lower()


# ============================================================
# 5. REMOVE URLs
# ============================================================

df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        x
    )
)


# ============================================================
# 6. REMOVE EMAIL ADDRESSES
# ============================================================

df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(
        r"\S+@\S+",
        " ",
        x
    )
)


# ============================================================
# 7. REMOVE EMOJIS / EMOTICONS
# ============================================================

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE
)

df["clean_text"] = df["clean_text"].apply(
    lambda x: emoji_pattern.sub(" ", x)
)


# ============================================================
# 8. NORMALIZE NON-ALPHABETIC CHARACTERS
# ============================================================

# Replace punctuation, numbers and special characters with spaces.
df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(
        r"[^a-zA-Z]+",
        " ",
        x
    )
)

# Remove multiple spaces
df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(
        r"\s+",
        " ",
        x
    ).strip()
)


# ============================================================
# 9. TOKENIZATION
# ============================================================

tokenizer = RegexpTokenizer(r"[a-zA-Z]+")

df["tokens"] = df["clean_text"].apply(
    tokenizer.tokenize
)


# ============================================================
# 10. REMOVE STOPWORDS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

df["tokens"] = df["tokens"].apply(
    lambda tokens: [
        word
        for word in tokens
        if word not in stop_words
    ]
)


# ============================================================
# 11. LEMMATIZATION
# ============================================================

lemmatizer = WordNetLemmatizer()

df["tokens"] = df["tokens"].apply(
    lambda tokens: [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]
)


# ============================================================
# 12. JOIN TOKENS BACK INTO TEXT
# ============================================================

df["clean_text"] = df["tokens"].apply(
    lambda tokens: " ".join(tokens)
)


# ============================================================
# 13. REMOVE EMPTY TEXT RECORDS
# ============================================================

before_empty_removal = len(df)

df = df[
    df["clean_text"].str.strip() != ""
].reset_index(drop=True)

removed_empty = (
    before_empty_removal - len(df)
)

print("\n========== EMPTY RECORD REMOVAL ==========")
print("Empty records removed:", removed_empty)


# ============================================================
# 14. SAVE PREPROCESSED DATASET
# ============================================================

output_path = "data/pirina_reddit_preprocessed.csv"

df[
    [
        "text",
        "clean_text",
        "label",
        "source",
        "filename"
    ]
].to_csv(
    output_path,
    index=False
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n========== PREPROCESSING COMPLETE ==========")

print("Final shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df[
    ["text", "clean_text", "label"]
].isnull().sum())


# ============================================================
# 16. BEFORE / AFTER EXAMPLES
# ============================================================

print("\n========== SAMPLE TEXTS ==========")

for i in range(min(3, len(df))):

    print(f"\n--- Sample {i + 1} ---")

    print("\nOriginal:")
    print(df.loc[i, "text"][:400])

    print("\nCleaned:")
    print(df.loc[i, "clean_text"][:400])

    print("\nLabel:")
    print(df.loc[i, "label"])


print("\n======================================")
print("Saved file:", output_path)
print("======================================")