import streamlit as st
import pickle
import numpy as np
import re
import nltk
import ftfy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
nltk.download("stopwords", quiet=True)

# ─────────────────────────────────────────────────────
# Text Cleaner
# ─────────────────────────────────────────────────────
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
            "]+", flags=re.UNICODE
        )
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

# ─────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# ─────────────────────────────────────────────────────
# Keyword Extraction
# ─────────────────────────────────────────────────────
def get_keywords(text, pipeline, n=5):
    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]
    cleaner    = pipeline.named_steps["cleaner"]

    cleaned = cleaner.transform([text])[0]
    tfidf_vec = vectorizer.transform([cleaned])

    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores  = tfidf_vec.toarray()[0]
    coef          = classifier.coef_[0]

    contributions = tfidf_scores * coef
    nonzero = np.where(tfidf_scores > 0)[0]

    if len(nonzero) == 0:
        return [], []

    words   = feature_names[nonzero]
    contrib = contributions[nonzero]

    pos_idx = np.argsort(contrib)[-n:]
    neg_idx = np.argsort(contrib)[:n]

    pos_kws = [words[i] for i in pos_idx if contrib[i] > 0][::-1]
    neg_kws = [words[i] for i in neg_idx if contrib[i] < 0]

    return pos_kws, neg_kws

# ─────────────────────────────────────────────────────
# Highlight Function
# ─────────────────────────────────────────────────────
def highlight_text(text, pos_words, neg_words):
    for w in pos_words:
        text = re.sub(rf"\b({w})\b",
                      r"<span style='color:green;font-weight:bold'>\1</span>",
                      text, flags=re.IGNORECASE)

    for w in neg_words:
        text = re.sub(rf"\b({w})\b",
                      r"<span style='color:red;font-weight:bold'>\1</span>",
                      text, flags=re.IGNORECASE)
    return text

# ─────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────
st.title("🎬 IMDB Sentiment Analyzer")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_keywords = st.checkbox("Show keywords", True)
    show_cleaned  = st.checkbox("Show cleaned text", False)

# Input
text = st.text_area(
    "Write a review:",
    placeholder="This movie was absolutely amazing...",
    height=150
)

# ─────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Analyzing..."):
        try:
            proba = pipeline.predict_proba([text])[0]
            pred  = pipeline.predict([text])[0]
        except Exception:
            st.error("An error occurred while running the model.")
            st.stop()

    pos_conf = float(proba[1])
    neg_conf = float(proba[0])
    is_pos   = pos_conf > neg_conf

    # Result
    if is_pos:
        st.success(f"😊 Positive — {pos_conf*100:.1f}% confidence")
    else:
        st.error(f"😔 Negative — {neg_conf*100:.1f}% confidence")

    # Confidence
    st.subheader("📊 Confidence Scores")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive", f"{pos_conf*100:.1f}%")
    with col2:
        st.metric("Negative", f"{neg_conf*100:.1f}%")

    st.progress(pos_conf)

    # Cleaned text
    cleaned_text = pipeline.named_steps["cleaner"].transform([text])[0]

    if show_cleaned:
        st.subheader("🧹 Cleaned Text")
        st.code(cleaned_text)

    # Keywords
    pos_kws, neg_kws = get_keywords(text, pipeline)

    if show_keywords:
        st.subheader("🧠 What Did the Model Base Its Decision On?")
        highlighted = highlight_text(text, pos_kws, neg_kws)
        st.markdown(highlighted, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("🟢 Positive keywords")
            for w in pos_kws:
                st.write(f"• {w}")
        with col2:
            st.write("🔴 Negative keywords")
            for w in neg_kws:
                st.write(f"• {w}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | IMDB Dataset")
