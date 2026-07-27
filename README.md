# 🌾 Causal Crop Advisor

A Streamlit-based decision support system that helps farmers identify the agricultural factors that truly influence crop yield using **Causal Inference** and **Machine Learning**.

---

## 📌 Overview

Traditional crop prediction systems often rely only on correlations between variables and yield.

This application adopts a **causal inference approach**, identifying the variables that have a genuine causal influence on crop yield before using machine learning models for prediction.

The goal is to provide:

- Better interpretability
- Actionable recommendations
- Simpler decision making
- Reliable yield estimation

---

## ✨ Features

- 🌾 Crop Yield Prediction
- 📊 Causal Impact Dashboard
- 🧠 Yield Explanation
- 👨‍🌾 Farmer Recommendations
- 📄 Downloadable Farmer Report
- ⚡ Interactive Streamlit Interface

---

## 🔬 Methodology

The workflow followed in this application is:

```
Field Inputs
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

Instead of relying on all available variables, the prediction is based on a **causally selected subset of agricultural features**.

---

## 🌱 Causally Important Variables

The deployed model focuses on the following variables:

- Average Soil Moisture
- Average Temperature
- Solar Radiation
- Climatic Water Balance
- NDVI
- FPAR
- Available Water Capacity
- Bulk Density
- Drainage Class

These variables were identified through causal analysis before model development.

---

## 🤖 Models Used

Three machine learning models are included:

- Random Forest
- XGBoost
- LightGBM

The application compares predictions from these models and reports the final estimated crop yield.

---

## 🛠 Technology Stack

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- ReportLab

---

## 📂 Project Structure

```
Causal-Crop-Advisor
│
├── app.py
├── assets/
├── data/
│   └── models/
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

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

## 📷 Application

The application allows farmers to:

- Enter field conditions
- View causally important factors
- Estimate crop yield
- Receive field-specific recommendations
- Generate a farmer report

---

## 📚 Related Research

The complete research implementation, including:

- Exploratory Data Analysis
- Causal Discovery
- Treatment Effect Estimation
- Feature Selection
- Model Development

is available in the companion repository:

**Causal-Crop-Yield-Forecasting**

---

## 👨‍💻 Developer

**Aditya Nagsen**

M.Sc. Applied Statistics (Data Science)

Symbiosis Statistical Institute, Pune

---

## 📄 License

This project is intended for academic and educational purposes.