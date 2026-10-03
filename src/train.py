import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import preprocess_text

train_df = pd.read_csv("Dataset/train.csv")
train_df = preprocess_text(train_df)

X = train_df["text"]
y = train_df["Class Index"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=100000, ngram_range=(1, 2), sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/news_classifier.pkl")

print("Model training completed.")
print("Model saved at: model/news_classifier.pkl")