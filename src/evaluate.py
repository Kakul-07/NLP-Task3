import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess_text

test_df = pd.read_csv("Dataset/test.csv")
test_df = preprocess_text(test_df)

X_test = test_df["text"]
y_test = test_df["Class Index"]

model = joblib.load("model/news_classifier.pkl")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["World", "Sports", "Business", "Sci/Tech"]))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["World", "Sports", "Business", "Sci/Tech"]
)

display.plot()
plt.title("News Classification Confusion Matrix")
plt.tight_layout()
plt.show()

test_df["Predicted Class"] = y_pred
test_df.to_csv("Dataset/test_predictions.csv", index=False)

print("\nPredictions saved to Dataset/test_predictions.csv")