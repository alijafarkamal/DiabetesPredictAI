import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load diabetes prediction objects
try:
    impute_means = joblib.load('impute_means.pkl')
    scaler = joblib.load('scaler.pkl')
    loaded_model = joblib.load('model.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError as e:
    st.error(f"Error: Missing file {str(e)}. Please run diabetes_prediction.py to generate model files.")
    st.stop()

# Define columns to impute
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def diabetes_prediction(input_data):
    input_data = list(input_data)
    for i, col in enumerate(feature_names):
        if col in columns_to_impute and input_data[i] == 0:
            input_data[i] = impute_means[col]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    std_data = scaler.transform(input_df)
    prediction = loaded_model.predict(std_data)
    return 'The person is not diabetic' if prediction[0] == 0 else 'The person is diabetic'

# Custom CSS for healthcare-friendly styling
st.markdown("""
    <style>
    .main-header {
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #34495e;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 16px;
        color: #2c3e50;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .prediction-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stNumberInput input {
        border-radius: 5px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">Diabetes Prediction AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Welcome! I’m an AI agent that predicts diabetes risk based on medical data. Enter your details below to get started.</div>', unsafe_allow_html=True)

    # Sidebar with additional information
    st.sidebar.title("About")
    st.sidebar.write("""
    This AI agent uses a Support Vector Machine (SVM) model trained on the PIMA Indians Diabetes Dataset to predict diabetes risk.
    
    The model was trained with the following steps:
    1. Data preprocessing and standardization.
    2. Train-test splitting.
    3. Model training with a linear kernel SVM.
    4. Evaluation using accuracy scores.
    
    For more information, visit the [GitHub repository](https://github.com/zohaib-7035/Diabetes-Prediction-Using-SVM-in-Python).
    """)

    # Input fields with descriptions and tooltips
    st.markdown('<div class="sub-header">Enter Your Medical Data</div>', unsafe_allow_html=True)
    st.write("Please provide the following information. Zeros in certain fields will be imputed with average values.")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, help="Number of times pregnant (0-20)")
        glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=300, value=0, help="Blood sugar level (typical range: 70-140 mg/dL)")
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=0, help="Diastolic blood pressure (typical range: 60-90 mm Hg)")
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=0, help="Triceps skin fold thickness (mm)")

    with col2:
        insulin = st.number_input('Insulin (mu U/ml)', min_value=0, max_value=1000, value=0, help="2-hour serum insulin (mu U/ml)")
        bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0, help="Body Mass Index (typical range: 18.5-30)")
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.0, help="Genetic diabetes risk score (0.0-3.0)")
        age = st.number_input('Age', min_value=0, max_value=120, value=0, help="Age in years (0-120)")

    # Collect inputs
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

    # Predict button with loading spinner
    if st.button('Predict'):
        if any(val < 0 for val in input_data):
            st.error("Please enter non-negative values for all fields.")
        else:
            with st.spinner('Predicting...'):
                diagnosis = diabetes_prediction(input_data)
                if 'not diabetic' in diagnosis.lower():
                    st.markdown(f'<div class="prediction-success">{diagnosis}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-error">{diagnosis}</div>', unsafe_allow_html=True)

    # Feature explanations expander
    with st.expander("Learn More About the Features"):
        st.write("""
        - **Pregnancies**: Number of times pregnant.
        - **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
        - **Blood Pressure**: Diastolic blood pressure (mm Hg).
        - **Skin Thickness**: Triceps skin fold thickness (mm).
        - **Insulin**: 2-Hour serum insulin (mu U/ml).
        - **BMI**: Body mass index (weight in kg/(height in m)^2).
        - **Diabetes Pedigree Function**: A function that scores likelihood of diabetes based on family history.
        - **Age**: Age in years.
        """)

    # Disclaimer
    st.markdown('<div class="info-box">*Disclaimer: This prediction is not a substitute for professional medical advice. Consult a healthcare provider for an accurate diagnosis.*</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()