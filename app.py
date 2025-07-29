
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Employee Pay Predictor", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K using demographic data.")

# Sidebar inputs
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", encoder.classes_)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=200000)
educational_num = st.sidebar.slider("Educational Number", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", encoder.classes_)
occupation = st.sidebar.selectbox("Occupation", encoder.classes_)
relationship = st.sidebar.selectbox("Relationship", encoder.classes_)
race = st.sidebar.selectbox("Race", encoder.classes_)
gender = st.sidebar.selectbox("Gender", encoder.classes_)
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", encoder.classes_)

# Prepare input
input_data = pd.DataFrame([{
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}])

# Encoding
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols:
    input_data[col] = encoder.transform(input_data[col])

# Scaling
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_scaled)
    result = "Salary >50K" if prediction[0] == ">50K" else "Salary ≤50K"
    st.success(f"Prediction: {result}")

# Batch Prediction
st.markdown("---")
st.markdown("### 📂 Batch Prediction from CSV")
batch_file = st.file_uploader("Upload a CSV file", type="csv")

if batch_file:
    batch_data = pd.read_csv(batch_file)
    for col in categorical_cols:
        batch_data[col] = encoder.transform(batch_data[col])
    batch_scaled = scaler.transform(batch_data)
    preds = model.predict(batch_scaled)
    batch_data['Predicted Salary'] = preds
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Result CSV", csv, "salary_predictions.csv", "text/csv")
