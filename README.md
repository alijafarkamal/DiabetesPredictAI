# DiabetesPredictAI – Agentic AI for Risk Assessment & Education  
**Traversaal x Optimized AI Hackathon**

DiabetesPredictAI is a proof-of-concept **agentic AI system** designed to assess diabetes risk, provide precautionary recommendations, and retrieve relevant educational content. Built with **Streamlit** and powered by **AgentPro** – [Traversaal’s open-source, production-ready agent framework](https://github.com/traversaal-ai/AgentPro) – this project showcases the potential of modular AI agents in healthcare applications.

---

## 🧠 Project Overview

### 🎯 Purpose
Make **early diabetes risk prediction** and **preventive education** accessible and interactive for everyone.

### 💡 Core Features
- **Predict** diabetes risk using a trained **SVM model** on the **PIMA Indians Diabetes Dataset**.
- **Inform** users through intelligent answers to diabetes-related queries via **Ares API**.
- **Educate** users with curated YouTube videos tailored to their predicted risk profile.

### 🛠️ Technologies
`Python`, `Streamlit`, `AgentPro`, `Scikit-learn (SVM)`, `Ares API`, `YouTube Data API`

---

## Our Team:
- [Ali Jafar](https://github.com/alijafarkamal)
- [Zohaib Shahid](https://github.com/zohaib-7035)
- [Noor Fatima](https://github.com/Noor-Fatima-Afzal)
- [Haris Mughal](https://github.com/Haris-Mughal)

---

## ⚙️ Setup Instructions

### 🔐 Prerequisites
- Python 3.8 or higher
- Git

### 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/alijafarkamal/Traversaal-x-Optimized-AI-Hackathon.git
   cd Traversaal-x-Optimized-AI-Hackathon
   ```

2. **(Optional) Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r agentpro/requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root and add:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key
   TRAVERSAAL_ARES_API_KEY=your_traversaal_ares_api_key
   ```

5. **Ensure Data Files Exist in the Root Directory**
   - `diabetes.csv`
   - `feature_names.pkl`
   - `impute_means.pkl`
   - `model.pkl`
   - `scaler.pkl`
   - `trained_model.sav`

6. **Run the Application**
   ```bash
   streamlit run system.py
   ```

---

## 🧬 Our Approach

### 🩺 Problem
Millions remain undiagnosed or unaware of diabetes risk. Risk tools are often inaccessible or complex.

### 💡 Solution
Build an intelligent, web-based assistant to:
- **Predict** diabetes risk using user medical data.
- **Inform** through real-time search and general reasoning.
- **Educate** via curated, context-aware video recommendations.

---

## 🤖 Agentic Architecture Diagram

```plaintext
Streamlit UI
    |
    v
AgentPro (Main Agent)
    |
    +--------------------------+--------------------------+---------------------------+
    |                          |                          |                           |
DiabetesPredictionTool   AresInternetTool        YouTubeSearchTool
    |                          |                          |
SVM Model           Real-Time Search             Video Recommendations
```

---

## 📂 Project Structure

```
Traversaal-x-Optimized-AI-Hackathon
│
├── Diabetes_Prediction.ipynb
├── diabetes.csv
├── diabetes_prediction.py
├── diabetes_tool.py
├── feature_names.pkl
├── impute_means.pkl
├── model.pkl
├── model.py
├── scaler.pkl
├── system.py
├── trained_model.sav
├── user_data.csv
├── requirements.txt
├── .env
│
└── agentpro/
    ├── requirements.txt
    └── agentpro/
        ├── agent.py
        ├── __init__.py
        ├── tools/
        │   ├── diabetes_tool.py
        │   ├── youtube_tool.py
        │   └── ares_tool.py
        └── examples/
            ├── Quick_Start.ipynb
            └── Custool_Tool_Integration.ipynb
```

---

## 🧩 Why AgentPro?

We chose AgentPro for its robust agentic framework enabling:
- **Tool Integration**: Multiple agents (prediction, search, video) working in harmony.
- **Context-Aware Routing**: Intelligent delegation of user queries to the right agent.
- **Extensibility**: Easily add new tools or expand functionality with minimal changes.

---

## 🌟 Bonus Points

✅ Built entirely on **AgentPro**  
✅ Modular, agentic AI architecture  
✅ Production-ready tool integration  
✅ Simple yet powerful Streamlit interface  
✅ Personalized outputs + educational impact

---

## 🔗 Repository

GitHub: [Traversaal x Optimized AI Hackathon](https://github.com/alijafarkamal/Traversaal-x-Optimized-AI-Hackathon)

---
