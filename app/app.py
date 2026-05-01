import streamlit as st
import pandas as pd
import joblib
import os
from collections import Counter
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# -------------------------------
# FORCE DARK THEME + FIX UI
# -------------------------------
st.markdown("""
<style>

/* Remove top white space */
.block-container {
    padding-top: 1rem !important;
}

/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Title */
h1 {
    text-align: center;
    color: white !important;
}

/* Subtext */
p {
    text-align: center;
    color: #cbd5f5;
}

/* Text area */
textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #475569 !important;
}

/* Input labels */
label {
    color: white !important;
    font-weight: bold;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

/* KPI Cards */
.kpi {
    background: #1e293b;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #1e293b;
    color: white;
}

/* Remove white header area */
header, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model_path = os.path.join("models", "model.pkl")
vectorizer_path = os.path.join("models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Sentiment Analysis Dashboard")
st.markdown("Analyze customer sentiment from social media text")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Controls")
option = st.sidebar.radio("Choose Input Type", ["Single Text", "Upload CSV"])

# -------------------------------
# FUNCTIONS
# -------------------------------
def predict(texts):
    X = vectorizer.transform(texts)
    return model.predict(X)

def top_words(texts):
    words = " ".join(texts).split()
    return pd.DataFrame(Counter(words).most_common(10), columns=["Word", "Count"])

# -------------------------------
# SINGLE TEXT
# -------------------------------
if option == "Single Text":
    text = st.text_area("Enter your text:")

    if st.button("Analyze"):
        pred = predict([text])[0]

        st.success(f"Prediction: {pred.upper()}")

# -------------------------------
# CSV INPUT
# -------------------------------
else:
    file = st.file_uploader("Upload CSV (must have 'text' column)", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            df["Predicted"] = predict(df["text"])

            st.success("Analysis Complete")

            st.dataframe(df.head())

            # -------------------------------
            # KPIs
            # -------------------------------
            total = len(df)
            pos = (df["Predicted"] == "positive").sum()
            neg = (df["Predicted"] == "negative").sum()
            neu = (df["Predicted"] == "neutral").sum()

            st.markdown("## 📌 Key Metrics")

            col1, col2, col3, col4 = st.columns(4)

            col1.markdown(f"<div class='kpi'><h3>Total</h3><h2>{total}</h2></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='kpi'><h3>Positive 😊</h3><h2 style='color:#22c55e'>{pos}</h2></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='kpi'><h3>Negative 😡</h3><h2 style='color:#ef4444'>{neg}</h2></div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='kpi'><h3>Neutral 😐</h3><h2 style='color:#facc15'>{neu}</h2></div>", unsafe_allow_html=True)

            # -------------------------------
            # CHARTS
            # -------------------------------
            st.markdown("## 📊 Visual Insights")

            col1, col2 = st.columns(2)

            fig1 = px.pie(df, names="Predicted", title="Sentiment Distribution")
            fig2 = px.bar(df["Predicted"].value_counts(), title="Sentiment Count")

            col1.plotly_chart(fig1, use_container_width=True)
            col2.plotly_chart(fig2, use_container_width=True)

            # -------------------------------
            # TOP WORDS
            # -------------------------------
            st.markdown("## 🔤 Top Words")

            c1, c2, c3 = st.columns(3)

            c1.dataframe(top_words(df[df["Predicted"]=="positive"]["text"]))
            c2.dataframe(top_words(df[df["Predicted"]=="negative"]["text"]))
            c3.dataframe(top_words(df[df["Predicted"]=="neutral"]["text"]))