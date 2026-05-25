import re
import nltk
import ftfy
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

nltk.download("stopwords", quiet=True)

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=True, remove_emoji=True):
        self.remove_stopwords = remove_stopwords
        self.remove_emoji = remove_emoji

    def fit(self, X, y=None):
        stopwords_set = set(stopwords.words("english"))
        negations = {"not", "no", "never", "neither", "nor", "hardly", "barely"}
        self.stopwords = stopwords_set - negations
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "]+", flags=re.UNICODE)
        return self

    def transform(self, X, y=None):
        return [self._clean(t) for t in X]

    def _clean(self, text):
        text = ftfy.fix_text(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        if self.remove_emoji:
            text = self._emoji_pattern.sub("", text)
        text = re.sub(r"([!?.]){2,}", r"\1", text)
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = text.lower()
        if self.remove_stopwords:
            tokens = [t for t in text.split() if t not in self.stopwords]
            text = " ".join(tokens)
        return re.sub(r"\s+", " ", text).strip()

# ── Dataset ───────────────────────────────────────────────────────────────────
df = pd.read_csv("IMDB Dataset.csv")
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y)

# ── Eğit ──────────────────────────────────────────────────────────────────────
full_pipeline = Pipeline([
    ("cleaner",    TextCleaner(remove_stopwords=True, remove_emoji=True)),
    ("vectorizer", TfidfVectorizer(max_features=1000, ngram_range=(1,2), sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ── Kaydet ────────────────────────────────────────────────────────────────────
with open("pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)

print("pipeline.pkl kaydedildi ✅")
