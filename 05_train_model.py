import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1. Load dataset
# =========================

data = pd.read_csv("data/sign_data.csv")

print("Dataset loaded!")
print("Total samples:", len(data))
print("Signs found:", data["label"].unique())


# =========================
# 2. Separate features/labels
# =========================

X = data.drop("label", axis=1)
y = data["label"]


# =========================
# 3. Split dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 4. Create ML model
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# =========================
# 5. Train model
# =========================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete!")


# =========================
# 6. Test model
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n=========================")
print("MODEL RESULTS")
print("=========================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print("Labels:", model.classes_)
print(cm)


# =========================
# 7. Save model
# =========================

model_path = "models/sign_model.pkl"

import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)
