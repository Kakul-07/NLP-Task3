# News Classification System

This project is a machine learning project that predicts the category of a news article using its title and description.

The news is classified into four categories:

1. World
2. Sports
3. Business
4. Sci/Tech

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- FastAPI
- Uvicorn
- Matplotlib

## Dataset

The dataset contains news articles with three main columns:

- Title
- Description
- Class Index

The class index is:

| Class Index | Category |
|---|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

## How It Works

First, the title and description are combined.

The text is then cleaned by converting it to lowercase, removing unwanted characters and extra spaces.

TF-IDF is used to convert the text into numerical features.

Logistic Regression is then used to predict the category.

The trained model is saved as `news_classifier.pkl` inside the `model` folder.

## Result

The model achieved around 92% accuracy on the test dataset.

A classification report and confusion matrix were also used to check the model performance.

## How to Run

Install the required libraries:

python3 -m pip install -r requirements.txt

Train the model:

python3 src/train.py

Evaluate the model:

python3 src/evaluate.py

Run the API:

python3 -m uvicorn src.app:app --reload --port 8001

Open the API documentation:

http://127.0.0.1:8001/docs

## API Example

For the `/predict` endpoint, send:

{
    "title": "India wins the cricket match",
    "description": "India won the match after a close game."
}

Example response:

{
    "prediction": "Sports"
}

## Main Files

`preprocessing.py` - cleans and prepares the news text.

`train.py` - trains the machine learning model and saves it.

`evaluate.py` - checks the model performance on the test data.

`app.py` - runs the FastAPI and makes predictions.

`eda.ipynb` - used to explore the dataset.

## Live API

The FastAPI application is deployed on Render.

API:
https://nlp-task3.onrender.com

API Documentation:
https://nlp-task3.onrender.com/docs