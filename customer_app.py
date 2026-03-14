import streamlit as st
import joblib
import re
from sentence_transformers import SentenceTransformer

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="📋",
    layout="centered"
)

st.title("📋 Customer Complaint Classification System")
st.write("Enter a customer complaint below to identify the corresponding category.")

# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text.strip()

# LOAD MODELS
@st.cache_resource
def load_models():
    model = joblib.load("customer_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    return model,encoder,bert

model,le,bert_model = load_models()

# USER INPUT
st.subheader("Enter Customer Complaint")

user_notice = st.text_area(
    "Customer Complaint Text",
    height=200,
    placeholder="Type or paste a customer complaint here..."
)

# PREDICTION
if st.button("Predict Complaint Category"):

    if user_notice.strip()=="":
        st.warning("Please enter a customer complaint.")

    else:
        cleaned_complaint = clean_text(user_notice)

        complaint_vector = bert_model.encode([cleaned_complaint])

        prediction = model.predict(complaint_vector)

        category_name = le.inverse_transform(prediction)

        st.success(f"Predicted Complaint Category: **{category_name[0]}**")

# FOOTER
st.markdown("---")
st.caption("Customer Complaint Classification")