
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Employee Pay Predictor", layout="wide")
st.title("Employee Pay Predictor")

st.write("Upload your dataset (CSV format) or use the default dataset.")

uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
else:
    # Load default dataset
    df = pd.read_csv("adult 3.csv")  # This file must exist in the same directory
    st.warning("Using default dataset: 'adult 3.csv'")

# Display the dataset
st.subheader("Sample of Dataset")
st.dataframe(df.head())

# Label Encoding
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

# Features and target
X = df.drop('income', axis=1)
y = df['income']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# UI for prediction
st.subheader("Predict Employee Pay Category")

# Input form
form = st.form("prediction_form")
form_cols = X.columns.tolist()
user_input = []

for col in form_cols:
    col_min = int(df[col].min())
    col_max = int(df[col].max())
    user_val = form.slider(f"{col}", col_min, col_max, int((col_min + col_max) / 2))
    user_input.append(user_val)

submit = form.form_submit_button("Predict")

if submit:
    prediction = model.predict([user_input])[0]
    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Salary Category: {label}")
