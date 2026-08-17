import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

st.set_page_config(page_title="Sustainability Sentiment Analytics", page_icon="🌱", layout="wide")

@st.cache_resource
def get_vader():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()

sia = get_vader()

st.title("🌱 Sustainability Sentiment Analytics")
st.caption("Upload consumer sustainability comments to explore sentiment, model evaluation and LDA topics.")

def find_col(df, names):
    exact = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in exact:
            return exact[n.lower()]
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(n.lower() in lc for n in names):
            return c
    return None

def clean_text(x):
    if pd.isna(x): return ""
    x = str(x)
    x = re.sub(r"http\S+|www\S+|@\w+|<[^>]+>", " ", x)
    x = re.sub(r"#(\w+)", r"\1", x)
    return re.sub(r"\s+", " ", x).strip()

def vader_label(x):
    if x >= 0.05: return "Positive"
    if x <= -0.05: return "Negative"
    return "Neutral"

def norm_label(x):
    x = str(x).strip().lower()
    if x in {"positive","pos","1","1.0"}: return "Positive"
    if x in {"neutral","neu","0","0.0"}: return "Neutral"
    if x in {"negative","neg","-1","-1.0"}: return "Negative"
    return np.nan

def top_words(model, vectorizer, n=10):
    words = vectorizer.get_feature_names_out()
    rows = []
    for i, comp in enumerate(model.components_, 1):
        idx = comp.argsort()[:-n-1:-1]
        rows.append({"Topic": f"Topic {i}", "Top words": ", ".join(words[idx])})
    return pd.DataFrame(rows)

st.sidebar.header("Upload dataset")
file = st.sidebar.file_uploader("CSV or Excel", type=["csv","xlsx"])

if not file:
    st.info("Upload your dataset from the sidebar to begin.")
    st.markdown("""
**Required:** a text column such as `Comment`, `Text`, `Review` or `Feedback`.

**Optional:** `Sentiment`, `Source`, `Topic`, `Language`, `Date`.

If a labelled Sentiment column exists, the app calculates accuracy, precision, recall,
F1-score and a confusion matrix.
""")
    st.stop()

try:
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

comment_col = find_col(df, ["comment","comments","text","review","feedback","content"])
sent_col = find_col(df, ["sentiment","label","sentiment_label"])
source_col = find_col(df, ["source","platform","site"])
topic_col = find_col(df, ["topic","category","sustainability_topic"])
lang_col = find_col(df, ["language","lang"])
date_col = find_col(df, ["date","year","timestamp"])

if comment_col is None:
    st.error("No text/comment column was detected. Rename it to Comment or Text.")
    st.stop()

work = df.copy()
work["cleaned_comment"] = work[comment_col].apply(clean_text)
work = work[work["cleaned_comment"].str.len() > 0].copy()

with st.spinner("Running VADER..."):
    scores = work["cleaned_comment"].apply(sia.polarity_scores)
    work["vader_compound"] = scores.apply(lambda x: x["compound"])
    work["vader_positive"] = scores.apply(lambda x: x["pos"])
    work["vader_neutral"] = scores.apply(lambda x: x["neu"])
    work["vader_negative"] = scores.apply(lambda x: x["neg"])
    work["vader_sentiment"] = work["vader_compound"].apply(vader_label)

st.sidebar.header("Filters")
filtered = work.copy()

if source_col:
    vals = sorted(filtered[source_col].dropna().astype(str).unique())
    pick = st.sidebar.multiselect("Source", vals)
    if pick: filtered = filtered[filtered[source_col].astype(str).isin(pick)]

if topic_col:
    vals = sorted(filtered[topic_col].dropna().astype(str).unique())
    pick = st.sidebar.multiselect("Topic", vals)
    if pick: filtered = filtered[filtered[topic_col].astype(str).isin(pick)]

if lang_col:
    vals = sorted(filtered[lang_col].dropna().astype(str).unique())
    pick = st.sidebar.multiselect("Language", vals)
    if pick: filtered = filtered[filtered[lang_col].astype(str).isin(pick)]

if date_col:
    d = pd.to_datetime(filtered[date_col], errors="coerce")
    if d.notna().any():
        filtered = filtered.assign(_date=d)
        years = sorted(filtered["_date"].dropna().dt.year.unique())
        if years:
            yr = st.sidebar.multiselect("Year", years, default=years)
            if yr: filtered = filtered[filtered["_date"].dt.year.isin(yr)]

st.header("Dashboard Overview")
a,b,c,d = st.columns(4)
a.metric("Comments analysed", f"{len(filtered):,}")
b.metric("Positive", f"{(filtered.vader_sentiment=='Positive').sum():,}")
c.metric("Neutral", f"{(filtered.vader_sentiment=='Neutral').sum():,}")
d.metric("Negative", f"{(filtered.vader_sentiment=='Negative').sum():,}")

st.subheader("VADER Sentiment Distribution")
counts = filtered.vader_sentiment.value_counts().reindex(["Positive","Neutral","Negative"], fill_value=0)
fig, ax = plt.subplots(figsize=(8,4))
counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Sentiment"); ax.set_ylabel("Comments"); ax.set_title("VADER Sentiment Distribution")
plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

left,right = st.columns(2)
with left:
    if source_col:
        st.subheader("Sentiment by Source")
        tab = pd.crosstab(filtered[source_col].astype(str), filtered.vader_sentiment)
        tab = tab.reindex(columns=["Positive","Neutral","Negative"], fill_value=0)
        st.bar_chart(tab)
with right:
    if topic_col:
        st.subheader("Comments by Topic")
        st.bar_chart(filtered[topic_col].astype(str).value_counts())

if date_col and "_date" in filtered.columns:
    st.subheader("Sentiment Pattern Over Time")
    trend = pd.crosstab(filtered["_date"].dt.year, filtered.vader_sentiment)
    trend = trend.reindex(columns=["Positive","Neutral","Negative"], fill_value=0)
    st.line_chart(trend)

st.header("Model Evaluation")
if sent_col:
    ev = filtered.copy()
    ev["actual"] = ev[sent_col].apply(norm_label)
    ev = ev.dropna(subset=["actual"])
    if len(ev):
        y_true, y_pred = ev.actual, ev.vader_sentiment
        acc = accuracy_score(y_true, y_pred)
        p,r,f,s = precision_recall_fscore_support(
            y_true, y_pred, labels=["Positive","Neutral","Negative"], zero_division=0
        )
        x1,x2,x3,x4 = st.columns(4)
        x1.metric("Accuracy", f"{acc:.2%}")
        x2.metric("Macro Precision", f"{p.mean():.2%}")
        x3.metric("Macro Recall", f"{r.mean():.2%}")
        x4.metric("Macro F1", f"{f.mean():.2%}")
        st.dataframe(pd.DataFrame({
            "Sentiment":["Positive","Neutral","Negative"],
            "Precision":p, "Recall":r, "F1-score":f, "Support":s.astype(int)
        }), use_container_width=True)
        cm = confusion_matrix(y_true,y_pred,labels=["Positive","Neutral","Negative"])
        st.dataframe(pd.DataFrame(cm,
            index=["Actual Positive","Actual Neutral","Actual Negative"],
            columns=["Predicted Positive","Predicted Neutral","Predicted Negative"]),
            use_container_width=True)
    else:
        st.warning("A sentiment column was found, but no recognised labels were available.")
else:
    st.info("No manual sentiment labels were found, so model performance cannot be calculated.")

st.header("LDA Topic Modelling")
lda_df = filtered.copy()
if lang_col:
    lang = lda_df[lang_col].astype(str).str.strip().str.lower()
    mask = lang.isin(["us","en","english","eng","en-us","en_us"])
    if mask.any():
        lda_df = lda_df[mask].copy()
        st.caption(f"English-language subset: {len(lda_df):,} comments.")
    else:
        st.caption("No recognised English-language value found; current filtered data will be used.")

if len(lda_df) >= 20:
    topics = st.slider("Number of topics", 2, 10, 5)
    words = st.slider("Top words per topic", 5, 15, 10)
    if st.button("Run LDA", type="primary"):
        vec = CountVectorizer(max_df=.95, min_df=2, stop_words="english", max_features=3000)
        X = vec.fit_transform(lda_df.cleaned_comment)
        model = LatentDirichletAllocation(n_components=topics, random_state=42, learning_method="batch")
        doc_topic = model.fit_transform(X)
        lda_df["dominant_topic"] = doc_topic.argmax(axis=1)+1
        perp = model.perplexity(X)
        st.metric("LDA perplexity", f"{perp:,.2f}")
        tc = lda_df.dominant_topic.value_counts().sort_index()
        st.subheader("LDA Topic Distribution")
        st.bar_chart(tc)
        st.subheader("Top Words by Topic")
        st.dataframe(top_words(model,vec,words), use_container_width=True)
        st.download_button("Download LDA results",
            lda_df.to_csv(index=False).encode(), "lda_results.csv", "text/csv")
else:
    st.info("At least 20 usable comments are recommended for LDA.")

st.header("Download Analysed Results")
st.download_button(
    "Download VADER analysed CSV",
    filtered.drop(columns=["_date"], errors="ignore").to_csv(index=False).encode(),
    "sustainability_vader_analysed_results.csv", "text/csv"
)

st.divider()
st.caption("Exploratory prototype: automated outputs should support, not replace, professional judgement.")
