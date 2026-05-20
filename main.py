import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (StratifiedKFold, cross_val_score,train_test_split)
from sklearn.pipeline import make_pipeline
from collections import Counter
import re

st.set_page_config(
    page_title="SpamClassifier",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
:root{--bg:#0f1724;--card:#0b1220;--muted:#9aa4b2;--accent:#0ea5a4;--accent-2:#7c3aed}
*{box-sizing:border-box}
html,body,#root, .appview-container, .main, .block-container{background:linear-gradient(180deg,var(--bg),#071025) !important;color:#e6eef6}
.stApp {font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;}
.card{background:linear-gradient(180deg,var(--card), rgba(255,255,255,0.02));padding:18px;border-radius:12px;margin-bottom:16px;box-shadow:0 6px 18px rgba(2,6,23,0.6);}
.header-row{display:flex;align-items:center;gap:12px;flex-wrap:nowrap;white-space:nowrap}
.logo{width:64px;height:64px;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent-2));display:flex;align-items:center;justify-content:center;color:#021024;font-weight:800;font-size:26px;animation:float 4s ease-in-out infinite}
.lottie{width:72px;height:72px;animation:float 5s ease-in-out infinite;border-radius:8px}
.subtitle{color:var(--muted);margin-top:6px;display:block}
textarea{background:rgba(255,255,255,0.02) !important;color:inherit}
.stButton>button{background:linear-gradient(90deg,var(--accent),var(--accent-2)) !important;border:none !important;color:white !important}
.metric-label{color:var(--muted);font-weight:600}
.footer{color:var(--muted);font-size:13px;padding:14px 0;text-align:center}
@keyframes float{0%{transform:translateY(0)}50%{transform:translateY(-8px)}100%{transform:translateY(0)}}
@keyframes typing{from{width:0}to{width:100%}}
@media (max-width: 640px){.header-row{flex-direction:column;align-items:flex-start}}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

col_logo, col_title, col_anim = st.columns([1.2, 6, 2])
with col_logo:
    st.markdown("<div class='logo'>📩</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<div class='header-row'><div><h1 style='margin:0'>SpamClassifier</h1></div></div>", unsafe_allow_html=True)
with col_anim:
    # Embed a smaller Lottie animation to keep header on one line
    animation_html = """
    <html><body style='margin:0;background:transparent;'>
    <script src='https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js'></script>
    <div style='display:flex;align-items:center;justify-content:center;height:100%;'>
    <lottie-player class='lottie' src='https://assets9.lottiefiles.com/packages/lf20_tfb3estd.json' background='transparent' speed='1' loop autoplay style='width:72px;height:72px;'></lottie-player>
    </div>
    </body></html>
    """
    st.iframe(animation_html, height=120)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write('Upload your dataset or use the local `spam.csv` file. The app trains a lightweight Logistic Regression model for quick feedback.')
st.markdown("</div>", unsafe_allow_html=True)

ROOT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_DIR, "spam.csv")

def _read_raw_csv(path):
    if isinstance(path, (str, Path)):
        with open(path, encoding="latin-1") as f:
            return f.readlines()
    else:
        raw = path.read()
        if isinstance(raw, bytes):
            raw = raw.decode("latin-1")
        return raw.splitlines()


def _parse_label_message(lines):
    rows = []
    header = lines[0].strip().split(",", 1)
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split(",", 1)
        if len(parts) != 2:
            continue
        label, message = parts
        rows.append({"label": label.strip(), "message": message.strip()})
    return pd.DataFrame(rows)


def _load_csv_from_zip(source):
    with zipfile.ZipFile(source) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("Zip archive does not contain a CSV file")
        csv_names = sorted(csv_names, key=lambda name: (name.count("/"), name))
        with archive.open(csv_names[0]) as f:
            return pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")


@st.cache_data(show_spinner=False)
def load_data(path) -> pd.DataFrame:
    if isinstance(path, str) and not os.path.exists(path):
        return pd.DataFrame()

    try:
        if isinstance(path, str) and zipfile.is_zipfile(path):
            df = _load_csv_from_zip(path)
        elif hasattr(path, "name") and path.name.lower().endswith(".zip"):
            path.seek(0)
            df = _load_csv_from_zip(path)
        else:
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    except Exception:
        df = pd.DataFrame()

    if not df.empty and "v1" in df.columns and "v2" in df.columns:
        if df["v2"].isna().any() or df.shape[1] > 2:
            lines = _read_raw_csv(path)
            df = _parse_label_message(lines)
        else:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif not df.empty and "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]].rename(columns={"text": "message"})
    elif not df.empty and df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]
    else:
        try:
            lines = _read_raw_csv(path)
            df = _parse_label_message(lines)
        except Exception:
            return pd.DataFrame()

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


df = load_data(DATA_PATH)
if df.empty:
    st.warning("No valid local dataset found in the project root.")
    uploaded_file = st.file_uploader(
        "Upload your Kaggle spam dataset file",
        type=["csv", "zip"],
        help="Upload a .csv file or a .zip archive that contains the Kaggle spam dataset."
    )
    if uploaded_file is not None:
        df = load_data(uploaded_file)

if df.empty:
    st.error(
        "Dataset not found or invalid. Please place the Kaggle spam dataset file as `spam.csv` in the project root or upload it above."
    )
    st.info(
        "Example dataset format: columns `v1` and `v2` from the Kaggle spam dataset, where `v1` is label and `v2` is message text."
    )
    st.stop()

st.sidebar.header("Dataset summary")
st.sidebar.write(f"Total messages: {len(df):,}")
st.sidebar.write(f"Spam count: {int((df['label_num'] == 1).sum()):,}")
st.sidebar.write(f"Ham count: {int((df['label_num'] == 0).sum()):,}")

# Sidebar settings: dataset upload, retrain, theme & accent color
with st.sidebar:
    st.header("Settings")
    sb_uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"], help="Use v1/v2 or label/text format.")
    retrain_btn = st.button("Retrain model")
    st.write("---")
    st.write("Display")
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
    accent = st.color_picker("Accent color", "#0ea5a4")

# If user uploaded in sidebar, use that dataset
if 'sb_uploaded' in locals() and sb_uploaded is not None:
    df = load_data(sb_uploaded)
    if df.empty:
        st.error("Uploaded dataset is invalid. Please upload a valid Kaggle spam dataset CSV.")
        st.stop()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_model(df_in):
    if df_in.empty:
        raise ValueError("Cannot train model on empty dataset")

    df_in = df_in.copy()
    df_in["message"] = df_in["message"].astype(str).apply(clean_text)

    vc = df_in["label_num"].value_counts()
    n = len(df_in)
    n_classes = vc.size
    min_per_class = int(vc.min()) if not vc.empty else 0
    if n >= 8 and min_per_class >= 2:
        test_size_int = max(n_classes, int(round(n * 0.20)))
        if test_size_int < n:
            train_df, test_df = train_test_split(
                df_in,
                test_size=test_size_int,
                random_state=42,
                stratify=df_in["label_num"],
            )
        else:
            train_df = df_in.copy()
            test_df = df_in.iloc[0:0].copy()
    else:
        train_df = df_in.copy()
        test_df = df_in.iloc[0:0].copy()

    min_df = 2 if len(df_in) > 40 else 1
    vec_params = {
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": min_df,
        "max_df": 0.95,
        "sublinear_tf": True,
        "norm": "l2",
    }
    vec = TfidfVectorizer(**vec_params)
    X_train = vec.fit_transform(train_df["message"])
    y_train = train_df["label_num"]

    model = LogisticRegression(C=8.0, max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    if not test_df.empty:
        X_test = vec.transform(test_df["message"])
        y_test = test_df["label_num"]
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
    else:
        predictions = []
        y_test = pd.Series(dtype=int)
        accuracy = None
        if n >= 6 and min_per_class >= 2:
            n_splits = min(3, min_per_class)
            if n_splits >= 2:
                cv_pipeline = make_pipeline(TfidfVectorizer(**vec_params), LogisticRegression(C=8.0, max_iter=1000, solver="liblinear"))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(cv_pipeline, df_in["message"], df_in["label_num"], cv=cv, scoring="accuracy")
                accuracy = float(scores.mean())

    return model, vec, train_df, test_df, y_test, predictions, accuracy


# Train model once and store in session state; retrain if requested.
if "model" not in st.session_state or retrain_btn:
    model, vectorizer, train_df, test_df, y_test, predictions, accuracy = train_model(df)
    st.session_state["model"] = model
    st.session_state["vectorizer"] = vectorizer
    st.session_state["train_df"] = train_df
    st.session_state["test_df"] = test_df
    st.session_state["y_test"] = y_test
    st.session_state["predictions"] = predictions
    st.session_state["accuracy"] = accuracy
else:
    model = st.session_state["model"]
    vectorizer = st.session_state["vectorizer"]
    train_df = st.session_state["train_df"]
    test_df = st.session_state["test_df"]
    y_test = st.session_state["y_test"]
    predictions = st.session_state["predictions"]
    accuracy = st.session_state["accuracy"]


# Layout: messages + model performance
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Sample messages")
    st.dataframe(df.sample(min(12, len(df)), random_state=42).reset_index(drop=True))

    st.subheader("Top words per label")
    spam_text = " ".join(df[df["label_num"] == 1]["message"].astype(str).tolist())
    ham_text = " ".join(df[df["label_num"] == 0]["message"].astype(str).tolist())

    def top_n(text, n=10):
        tokens = [t.lower() for t in re.findall(r"\w+", text) if len(t) > 2]
        return [w for w, c in Counter(tokens).most_common(n)]

    st.markdown("**Spam:** " + ", ".join(top_n(spam_text, 10)))
    st.markdown("**Ham:** " + ", ".join(top_n(ham_text, 10)))

with col2:
    st.subheader("Model performance")
    acc_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
    st.metric("Test accuracy", acc_text)
    st.write("**Label distribution**")
    st.bar_chart(df["label"].value_counts())

with st.expander("Show dataset details"):
    st.dataframe(df["label"].value_counts().rename_axis("label").reset_index(name="count"))
    st.write("Columns:")
    st.write(df.columns.tolist())

st.markdown("---")

st.subheader("Predict scam email")
user_message = st.text_area("Enter email text", height=180)

if st.button("Predict"):
    if not user_message.strip():
        st.warning("Please enter the email text before predicting.")
    else:
        user_text = clean_text(user_message)
        vector = vectorizer.transform([user_text])
        prediction = model.predict(vector)[0]
        if prediction == 1:
            st.error("🚨 Spam/scam email detected")
        else:
            st.success("✅ Message appears clean")

with st.expander("Model evaluation"):
    if test_df.empty:
        st.info("Not enough data to evaluate model (no test set).")
    else:
        st.write("### Test set classification report")
        st.text(classification_report(y_test, predictions, target_names=["Ham", "Spam"]))
        st.write("### Example test predictions")
        sample_eval = test_df.copy()
        sample_eval = sample_eval.assign(
            actual=sample_eval["label"],
            predicted=pd.Series(predictions).map({0: "Ham", 1: "Spam"}),
        )[["message", "actual", "predicted"]]
        st.write(sample_eval.head(10))
