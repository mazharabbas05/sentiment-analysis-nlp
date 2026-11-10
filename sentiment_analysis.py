"""
Sentiment Analysis of Movie Reviews (NLP) - Mazhar Abbas

Pipeline:
  1. Load reviews (CSV with columns: review, sentiment)
  2. Clean the text (lowercase, remove HTML tags, punctuation, numbers)
  3. Split into train / test (80/20, stratified)
  4. Convert text to numbers with TF-IDF (unigrams + bigrams)
  5. Train and compare 3 models: Logistic Regression, Naive Bayes, Linear SVM
  6. Evaluate: accuracy, precision, recall, F1, confusion matrix
  7. Show the most influential positive / negative words

Usage:
    python sentiment_analysis.py "IMDB Dataset.csv"
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags like <br />
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text


def main(csv_path: str) -> None:
    # 1. Load data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["review", "sentiment"]).drop_duplicates()
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    print(f"Reviews used: {len(df)}")
    print(df["sentiment"].value_counts(), "\n")

    # 2. Clean text
    df["clean"] = df["review"].apply(clean_text)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42,
        stratify=df["label"])

    # 4. TF-IDF features
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                            max_features=50000, min_df=2)
    Xtr = tfidf.fit_transform(X_train)   # learn vocabulary on train only
    Xte = tfidf.transform(X_test)

    # 5. Train and compare models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }
    results, preds = {}, {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        p = model.predict(Xte)
        preds[name] = p
        results[name] = (accuracy_score(y_test, p), f1_score(y_test, p))
        print(f"{name:20s} accuracy={results[name][0]:.4f}  F1={results[name][1]:.4f}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\nBest model: {best}")
    print(classification_report(y_test, preds[best],
                                target_names=["negative", "positive"]))

    # 6. Confusion matrix of best model
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds[best], display_labels=["negative", "positive"],
        cmap="Blues")
    plt.title(f"Confusion Matrix - {best}")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Most influential words (from Logistic Regression)
    lr = models["Logistic Regression"]
    words = np.array(tfidf.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    print("Top negative words:", ", ".join(words[order[:15]]))
    print("Top positive words:", ", ".join(words[order[-15:][::-1]]))

    # Try your own review
    for text in ["This movie was absolutely wonderful, I loved it",
                 "Terrible plot and very boring, a waste of time"]:
        vec = tfidf.transform([clean_text(text)])
        label = "positive" if lr.predict(vec)[0] == 1 else "negative"
        print(f"'{text}' -> {label}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python sentiment_analysis.py "IMDB Dataset.csv"')
    main(sys.argv[1])
