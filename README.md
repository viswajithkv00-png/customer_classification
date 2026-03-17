# 📋 Customer Complaint Classification System

## 🌐 Live Demo

[Customer Complaint Classification System - Live App](https://customerclassification-gnbmenapptewhsjkbewgfng.streamlit.app/)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-green)](https://customerclassification-gnbmenapptewhsjkbewgfng.streamlit.app/)

A Streamlit-based Machine Learning application that classifies customer complaints into predefined categories using text embeddings and a trained scikit-learn model. This project is designed as a clean and practical NLP solution suitable for academic submission as well as a GitHub portfolio.

## Project Description

The **Customer Complaint Classification System** allows users to enter a customer complaint and instantly receive the predicted complaint category.  
It uses the **SentenceTransformer `all-MiniLM-L6-v2`** model to convert complaint text into embeddings, which are then passed to a trained **scikit-learn classifier** for prediction. The predicted numeric label is converted back to the original category using **LabelEncoder**.

## Features

- Clean and interactive **Streamlit web interface**
- Accepts free-text customer complaints as input
- Text preprocessing for cleaner prediction
- Uses **SentenceTransformer** for semantic text embeddings
- Predicts complaint category using a trained **Machine Learning model**
- Decodes output labels using **LabelEncoder**
- Beginner-friendly and easy to run locally

## Model Details

- **Embedding Model:** `all-MiniLM-L6-v2`
- **ML Model:** Trained scikit-learn classification model
- **Label Decoder:** `LabelEncoder`
- **Input:** Raw customer complaint text
- **Output:** Predicted complaint category

## Project Structure

```bash
Customer Complaint Classification System/
│
├── app.py
├── customer_app.py
├── train_customer.py
├── customer_model.pkl
├── label_encoder.pkl
├── customer_complaints.csv
├── requirement.txt
└── README.md
```

## Installation Steps

```bash
git clone <your-repository-url>
cd <project-folder>
pip install -r requirement.txt
```

## How to Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```bash
http://localhost:8501
```

## Example Input/Output

**Input Complaint**

```text
I was charged twice for my last purchase and the refund has not been processed yet.
```

**Predicted Output**

```text
Billing
```

## Author

**Your Name**  
Data Science / Machine Learning Project
