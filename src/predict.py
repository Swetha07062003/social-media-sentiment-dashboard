import joblib
import re

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

while True:
    text = input("Enter text: ")
    text = clean_text(text)
    vec = vectorizer.transform([text])
    print("Prediction:", model.predict(vec)[0])