from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    Path(r"D:\ML project\data\customer_complaints.csv"),
    BASE_DIR / "customer_complaints.csv",
]


def resolve_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("customer_complaints.csv was not found in any expected location.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data_path = resolve_data_path()
data = pd.read_csv(data_path)

if "text" in data.columns:
    data = data.rename(columns={"text": "complaint"})
if "label" in data.columns:
    data = data.rename(columns={"label": "category"})

print("Missing Values:")
print(data.isnull().sum())
print("\nDataset Preview:")
print(data.head())
print("\nDataset Tail:")
print(data.tail())
print("\nDataset Shape:")
print(data.shape)
print("\nDataset Info:")
data.info()

data = data.drop_duplicates().reset_index(drop=True)

plt.figure(figsize=(6, 4))
data["category"].value_counts().plot(kind="bar")
plt.title("Complaint Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Complaints")
plt.show()

data["complaint"] = data["complaint"].apply(clean_text)
data["length"] = data["complaint"].apply(len)

plt.figure(figsize=(6, 4))
sns.histplot(data["length"], bins=30)
plt.title("Complaint Length Distribution")
plt.xlabel("Length of Complaint")
plt.ylabel("Count")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    data["complaint"],
    data["category"],
    test_size=0.2,
    random_state=42,
    stratify=data["category"],
)

vectorizer = TfidfVectorizer(
    max_features=4000,
    stop_words="english",
    ngram_range=(1, 2),
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_names = vectorizer.get_feature_names_out()
importance = model.coef_[0]
top_words = pd.Series(importance, index=feature_names).sort_values(ascending=False)[:10]

print("\nTop Important Words:")
print(top_words)


def predict_complaint(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return le.inverse_transform(prediction)


print("\nComplaint Categories:")
print(le.classes_)
print(predict_complaint("refund not received"))
print(predict_complaint("internet not working"))
print(predict_complaint("order delivered late"))
print(predict_complaint("payment deducted but order failed"))
print(predict_complaint("wifi connection very slow"))
print(predict_complaint("product arrived damaged"))

joblib.dump(model, BASE_DIR / "complaint_classifier.pkl")
joblib.dump(vectorizer, BASE_DIR / "tfidf_vectorizer.pkl")
joblib.dump(le, BASE_DIR / "label_encoder.pkl")
