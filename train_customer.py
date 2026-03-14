import numpy as np
import pandas as pd
import re
import random
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# LOAD DATA
data = pd.read_csv("customer_complaints.csv")

# REMOVE DUPLICATES
data = data.drop_duplicates().reset_index(drop=True)

# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

data["text"] = data["text"].apply(clean_text)

# ADD COMMON CUSTOMER SUPPORT WORDS
common_words = [
    "issue","error","problem","complaint",
    "support","service","request","ticket",
    "urgent","resolve"
]

def add_common_words(text):
    return text + " " + " ".join(random.sample(common_words, 3))

data["text"] = data["text"].apply(add_common_words)

# ADD LABEL NOISE
noise_percentage = 0.05
num_noisy = int(len(data) * noise_percentage)

all_labels = list(data["label"].unique())
random_indices = random.sample(range(len(data)), num_noisy)

for idx in random_indices:
    current_label = data.loc[idx, "label"]
    other_labels = all_labels.copy()
    other_labels.remove(current_label)
    data.loc[idx, "label"] = random.choice(other_labels)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# LOAD BERT MODEL
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# BERT EMBEDDINGS
X_train = bert_model.encode(
    X_train.tolist(),
    batch_size=32,
    show_progress_bar=True
)

# LABEL ENCODING
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# TRAIN MODEL
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# SAVE MODEL
joblib.dump(model, "customer_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model training complete")
print("Classes:", le.classes_)
print("Model saved as customer_model.pkl")