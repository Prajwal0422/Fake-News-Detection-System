import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATASETS
# =========================

fake_df = pd.read_csv("../data/Fake.csv")
true_df = pd.read_csv("../data/True.csv")

# =========================
# ADD LABELS
# =========================

fake_df["label"] = 0
true_df["label"] = 1

# =========================
# COMBINE DATASETS
# =========================

df = pd.concat([fake_df, true_df])

# =========================
# KEEP REQUIRED COLUMNS
# =========================

df = df[["text", "label"]]

# =========================
# CHECK DATA
# =========================

print(df.head())

# =========================
# TEXT CLEANING FUNCTION
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text

# =========================
# APPLY CLEANING
# =========================

df["text"] = df["text"].apply(clean_text)

# =========================
# FEATURES AND TARGET
# =========================

X = df["text"]

y = df["label"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression()

model.fit(X_train_vec, y_train)

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test_vec)

# =========================
# ACCURACY
# =========================

acc = accuracy_score(y_test, y_pred) * 100

print(f"\nAccuracy: {acc:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("../outputs/confusion_matrix.png")

plt.show()

# =========================
# TEST CUSTOM NEWS
# =========================

news = input("\nEnter News Article:\n")

news_clean = clean_text(news)

news_vector = vectorizer.transform([news_clean])

prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")