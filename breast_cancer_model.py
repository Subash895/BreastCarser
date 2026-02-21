# ===============================
# Breast Cancer Prediction System
# ===============================

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# -------------------------------
# Data Collection & Processing
# -------------------------------

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(
    breast_cancer_dataset.data,
    columns=breast_cancer_dataset.feature_names
)

data_frame['label'] = breast_cancer_dataset.target

# Separate features and target
X = data_frame.drop(columns='label')
Y = data_frame['label']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# -------------------------------
# Model Training
# -------------------------------

model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# -------------------------------
# Model Evaluation
# -------------------------------

train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print("Training Accuracy :", train_accuracy)
print("Test Accuracy     :", test_accuracy)

# -------------------------------
# Save the Model
# -------------------------------

joblib.dump(model, "breast_cancer_model.pkl")
print("Model Saved Successfully!")

# -------------------------------
# Load the Model
# -------------------------------

loaded_model = joblib.load("breast_cancer_model.pkl")

# -------------------------------
# Predictive System (User Input)
# -------------------------------

print("\nEnter 30 feature values separated by comma:")
user_input = input()

# Convert string input into float list
input_data = list(map(float, user_input.split(",")))

if len(input_data) != 30:
    print("Error: You must enter exactly 30 values.")
else:
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = loaded_model.predict(input_df)

    if prediction[0] == 0:
        print("Result: The Breast Cancer is Malignant")
    else:
        print("Result: The Breast Cancer is Benign")