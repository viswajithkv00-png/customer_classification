from pathlib import Path
import re

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "complaint_classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


st.set_page_config(
    page_title="Customer Complaint Classification System",
    page_icon="\U0001F4CB",
    layout="centered",
)

st.title("Customer Complaint Classification System")
st.write("Enter a customer complaint below to identify the corresponding category.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, vectorizer, encoder


model, vectorizer, le = load_models()

st.subheader("Enter Customer Complaint")

user_notice = st.text_area(
    "Customer Complaint Text",
    height=200,
    placeholder="Type or paste a customer complaint here...",
)

if st.button("Predict Complaint Category"):
    if user_notice.strip() == "":
        st.warning("Please enter a customer complaint.")
    else:
        cleaned_complaint = clean_text(user_notice)
        complaint_vector = vectorizer.transform([cleaned_complaint])
        prediction = model.predict(complaint_vector)
        category_name = le.inverse_transform(prediction)[0]
        st.success(f"Predicted Complaint Category: **{category_name}**")

st.markdown("---")
st.caption("Customer Complaint Classification")
