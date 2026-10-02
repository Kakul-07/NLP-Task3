import re
def combine_text(df):
    df = df.copy()
    df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")
    return df
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def preprocess_text(df):
    df = combine_text(df)
    df["text"] = df["text"].apply(clean_text)
    return df