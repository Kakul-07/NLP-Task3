import joblib
import pandas as pd
from preprocessing import preprocess_text
model = joblib.load("model/news_classifier.pkl")
categories = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}
title = input("Enter news title: ")
description = input("Enter news description: ")
data = pd.DataFrame([
    {
        "Title": title,
        "Description": description
    }
])
data = preprocess_text(data)
prediction = model.predict(data["text"])[0]
print("\nPredicted Category:", categories[prediction])