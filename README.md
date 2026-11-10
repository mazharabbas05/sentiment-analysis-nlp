# Sentiment Analysis of Movie Reviews (NLP)

Classifies IMDB movie reviews as **positive** or **negative** using TF-IDF features and compares three machine learning models.

## Dataset
IMDB Dataset of 50K Movie Reviews (Kaggle, file `IMDB Dataset.csv`, columns `review` and `sentiment`).

## Approach
1. Clean text: lowercase, remove HTML tags, punctuation and numbers
2. Stratified 80/20 train-test split
3. TF-IDF features (unigrams + bigrams, English stop words removed, top 50,000 features)
4. Train and compare Logistic Regression, Multinomial Naive Bayes and Linear SVM
5. Evaluate with accuracy, precision, recall, F1 and a confusion matrix
6. Inspect the most influential positive and negative words

## Results
Fill in after running (copy from the terminal output):

| Model | Accuracy | F1-score |
|---|---|---|
| Logistic Regression | | |
| Naive Bayes | | |
| Linear SVM | | |

## How to run
```bash
pip install -r requirements.txt
python sentiment_analysis.py "IMDB Dataset.csv"
```
Output: model scores in the terminal and `confusion_matrix.png` in the same folder.

## Tech
Python, Pandas, NumPy, Scikit-learn, Matplotlib
