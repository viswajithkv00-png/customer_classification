from pathlib import Path
import os
import re

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "customer_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
SENTENCE_CACHE_DIR = BASE_DIR / ".cache" / "sentence_transformers"

os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_CACHE_DIR))


st.set_page_config(
    page_title="Customer Complaint Classification System",
    page_icon="\U0001F4CB",
    layout="centered",
)


def clean_text(text: str) -> str:
    """Normalize complaint text before generating embeddings."""
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource(show_spinner="Loading classifier and embedding model...")
def load_resources():
    """Load the trained classifier, label encoder, and embedding model once."""
    missing_files = [
        str(path.name) for path in (MODEL_PATH, ENCODER_PATH) if not path.exists()
    ]
    if missing_files:
        missing_names = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing required file(s): {missing_names}")

    from sentence_transformers import SentenceTransformer

    SENTENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    embedding_model = SentenceTransformer(
        SENTENCE_MODEL_NAME,
        device="cpu",
        cache_folder=str(SENTENCE_CACHE_DIR),
    )
    return model, label_encoder, embedding_model


def predict_category(complaint: str, model, label_encoder, embedding_model) -> str:
    """Convert complaint text into embeddings and return the predicted label."""
    cleaned_text = clean_text(complaint)
    complaint_embedding = embedding_model.encode(
        [cleaned_text],
        show_progress_bar=False,
    )
    predicted_label = model.predict(complaint_embedding)
    predicted_category = label_encoder.inverse_transform(predicted_label)
    return predicted_category[0]


def main():
    st.title("Customer Complaint Classification System")
    st.write(
        "Enter a customer complaint below to predict the most likely complaint category."
    )

    try:
        model, label_encoder, embedding_model = load_resources()
    except Exception as error:
        st.error(f"Unable to load the model files: {error}")
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
                complaint_text,
                model,
                label_encoder,
                embedding_model,
            )
            st.success(f"Predicted Complaint Category: {predicted_category}")

    st.markdown("---")
    st.caption("Built with Streamlit for customer complaint classification.")


if __name__ == "__main__":
    main()
