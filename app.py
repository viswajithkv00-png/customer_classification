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


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, vectorizer, label_encoder


def predict_category(complaint: str, model, vectorizer, label_encoder) -> str:
    cleaned_text = clean_text(complaint)
    complaint_vector = vectorizer.transform([cleaned_text])
    predicted_label = model.predict(complaint_vector)
    predicted_category = label_encoder.inverse_transform(predicted_label)
    return predicted_category[0]


def main():
    st.title("Customer Complaint Classification System")
    st.write(
        "Enter a customer complaint below to predict the most likely complaint category."
    )

    try:
        model, vectorizer, label_encoder = load_resources()
    except Exception as error:
        st.error(f"Unable to load the model files: {error}")
        st.info("Run `python train_customer.py` first to generate the required model files.")
        st.stop()

    complaint_text = st.text_area(
        "Customer Complaint",
        height=220,
        placeholder="Type or paste a customer complaint here...",
    )

    if st.button("Predict Complaint Category", use_container_width=True):
        if not complaint_text.strip():
            st.warning("Please enter a customer complaint before predicting.")
        else:
            predicted_category = predict_category(
                complaint_text, model, vectorizer, label_encoder
            )
            st.success(f"Predicted Complaint Category: {predicted_category}")

    st.markdown("---")
    st.caption("Built with Streamlit for customer complaint classification.")


if __name__ == "__main__":
    main()
