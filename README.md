# 🌾 Causal Crop Advisor

Helping farmers focus on the agricultural factors that **truly influence crop yield** using **Causal Inference** and **Machine Learning**.

## 🚀 Live Application

👉 **Launch the App:**  
**https://causal-crop-advisor.streamlit.app/**

---

## Overview

Traditional crop yield prediction systems often rely on every available variable, even when many are only correlated with crop yield.

This project adopts a **causal inference approach** to identify the variables that have a genuine causal influence on crop production before using machine learning models for prediction.

The application provides:

- 📊 Causal Impact Dashboard
- 🌾 Crop Yield Prediction
- 🧠 Explanation of the prediction
- 👨‍🌾 Actionable farmer recommendations
- 📄 Downloadable farmer report

---

## Workflow

```text
Field Information
        │
        ▼
Causal Variable Analysis
        │
        ▼
Machine Learning Prediction
        │
        ▼
Yield Explanation
        │
        ▼
Farmer Recommendations
```

---

## Causally Important Variables

The deployed model focuses on the following causally selected variables:

- 💧 Average Soil Moisture
- 🌡 Average Temperature
- ☀ Solar Radiation
- 🌦 Climatic Water Balance
- 🌿 NDVI
- 🌱 FPAR
- 🌍 Available Water Capacity
- 🪨 Bulk Density
- 🚜 Drainage Class

These variables were identified through causal discovery and treatment effect estimation before model development.

---

## Machine Learning Models

The application compares predictions from:

- Random Forest
- XGBoost
- LightGBM

The final recommendation is generated using the best-performing causal model.

---

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- Joblib
- ReportLab

---

## Repository Structure

```text
Causal-Crop-Advisor/
│
├── app.py
├── assets/
├── data/
│   └── models/
├── requirements.txt
└── README.md
```

---

## Running Locally

Clone the repository

```bash
git clone https://github.com/nagsen-aditya2611/Causal-Crop-Advisor.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Related Research

This application accompanies the research project:

**Crop Yield Forecasting using Causal Inference**

which includes:

- Exploratory Data Analysis
- Expert DAG Construction
- Causal Discovery
- Treatment Effect Estimation
- Causal Feature Selection
- Machine Learning Model Comparison
- Explainable AI

---

## Developer

**Aditya Nagsen**

M.Sc. Applied Statistics (Data Science)

Symbiosis Statistical Institute, Pune

---

## License

This project is intended for academic and educational purposes.