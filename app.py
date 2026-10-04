import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.preprocessing import preprocess_text

app = FastAPI(title="News Classification API")

model = joblib.load("model/news_classifier.pkl")

categories = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

class NewsInput(BaseModel):
    title: str
    description: str

@app.get("/")
def home():
    return {"message": "News Classification API is running"}

@app.post("/predict")
def predict(news: NewsInput):
    data = pd.DataFrame([
        {
            "Title": news.title,
            "Description": news.description
        }
    ])

    data = preprocess_text(data)

    prediction = model.predict(data["text"])[0]

    return {"prediction": categories[prediction]}
