import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.title("Employee Pay Predictor")

# Load CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(data.head())

    # Replace missing values
    data['occupation'].replace('?', 'Others', inplace=True)
    data['workclass'].replace('?', 'Notlisted', inplace=True)

    # Drop rows with specific unwanted values
    data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
    data = data[~data['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]

    # Drop redundant column
    data.drop(columns=['education'], inplace=True)

    # Remove outliers in age
    data = data[(data['age'] >= 17) & (data['age'] <= 75)]

    # Encode categorical variables
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Encode target variable
    target_encoder = LabelEncoder()
    data['income'] = target_encoder.fit_transform(data['income'])

    # Normalize features
    x = data.drop(columns=['income'])
    y = data['income']
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=1, stratify=y)

    # Classifier options
    classifier = st.selectbox("Choose a model", ["KNN", "Logistic Regression", "Neural Network"])

    if classifier == "KNN":
        model = KNeighborsClassifier()
    elif classifier == "Logistic Regression":
        model = LogisticRegression()
    elif classifier == "Neural Network":
        model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=2, max_iter=2000)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)

    st.subheader("Model Accuracy")
    st.write(f"{classifier} Accuracy: {acc:.2f}")

    # Display sample predictions
    st.subheader("Predictions")
    st.write(predictions[:10])