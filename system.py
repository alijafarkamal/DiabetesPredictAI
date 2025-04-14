import sys
import os
from dotenv import load_dotenv

# Load environment variables before any imports that depend on them
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agentpro')))
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from agentpro.agentpro import AgentPro, AresInternetTool, YouTubeSearchTool
from diabetes_tool import DiabetesPredictionTool


# Initialize tools and agent
try:
    diabetes_tool = DiabetesPredictionTool()
    internet_tool = AresInternetTool()
    youtube_tool = YouTubeSearchTool()

    agent = AgentPro(
        tools=[diabetes_tool, internet_tool, youtube_tool],
        model="nvidia/llama-3.1-nemotron-70b-instruct",  # Free model from OpenRouter
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        api_base="https://openrouter.ai/api/v1",  # OpenRouter API base
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
except Exception as e:
    st.error(f"Failed to initialize agent: {str(e)}")
    st.stop()

# Custom CSS for styling
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
    st.markdown('<div class="main-header">Diabetes AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Welcome! I’m an AI agent that can predict your diabetes risk, answer questions, and provide educational resources about diabetes. Ask me anything!</div>', unsafe_allow_html=True)

    # Sidebar with additional information
    st.sidebar.title("About")
    st.sidebar.write("""
    This AI agent uses a team of specialized agents to assist with diabetes-related queries:
    - Prediction Agent: Predicts diabetes risk using an SVM model trained on the PIMA Indians Diabetes Dataset.
    - Information Agent: Answers general questions about diabetes using internet search.
    - Educational Agent: Provides educational videos on diabetes management.
    
    For more information, visit the [GitHub repository](https://github.com/alijafarkamal/DiabetesPredictAI).
    """)

    # Input fields for medical data
    st.markdown('<div class="sub-header">Enter Your Medical Data (optional for predictions)</div>', unsafe_allow_html=True)
    st.write("Provide your details below. Zeros in certain fields will be imputed with average values.")
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

    if st.button('Set My Data'):
        st.session_state['medical_data'] = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        st.success("Your data has been set! You can now ask for a prediction.")

    # Chat interface
    st.markdown('<div class="sub-header">Ask Me Anything About Diabetes</div>', unsafe_allow_html=True)
    user_query = st.text_input("Your query:", placeholder="E.g., 'Predict my diabetes risk' or 'What are diabetes symptoms?'")

    if user_query:
        if 'medical_data' in st.session_state:
            query_with_data = f"{user_query} Use this data: {st.session_state['medical_data']}"
        else:
            query_with_data = user_query
        with st.spinner('Thinking...'):
            try:
                response = agent.run(query_with_data)
                if 'not diabetic' in response.lower():
                    st.markdown(f'<div class="prediction-success">{response}</div>', unsafe_allow_html=True)
                elif 'diabetic' in response.lower():
                    st.markdown(f'<div class="prediction-error">{response}</div>', unsafe_allow_html=True)
                else:
                    st.write(response)
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")

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