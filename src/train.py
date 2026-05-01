import pandas as pd
import os
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Ensure models folder exists
os.makedirs("../models", exist_ok=True)

print("Loading dataset...")
data = pd.read_csv("../data/dataset_10k.csv")

# Clean text
data['cleaned'] = data['text'].apply(clean_text)

# Split (IMPORTANT: stratify)
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned'],
    data['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=data['sentiment']
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save
joblib.dump(model, "../models/model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")
model = LogisticRegression(max_iter=200, C=0.5)
vectorizer = TfidfVectorizer(max_features=5000)

print("\n✅ Model and vectorizer saved successfully!")
print("\n🔍 Sample Data:\n")
print(data.head(10))

print("\n🔍 Unique samples:", len(data['text'].unique()))
print("Total samples:", len(data))