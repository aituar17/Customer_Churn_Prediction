# 💡 Customer_Churn_Prediction

## 📘 Overview
This project builds a **customer churn prediction model** using the Telco Customer Churn dataset.
It demonstrates the full data-science workflow — from **EDA and preprocessing**, through **modeling and evaluation**, to **explainability with SHAP**.

The focus is on **balancing accuracy with interpretability**, a key aspect in regulated industries such as **banking and risk analytics**.

## 🧩 Objectives
- Predict whether a telecom customer will churn (cancel service).
- Handle **class imbalance** using `class_weight = "balanced"`.
- Compare interpretable (Logistic Regression) and non-linear (Random Forest) models.
- Explain model behavior using **SHAP values** and **permutation importance**.

## 📊 Dataset
- **Source:** [Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 
- **File:** `data/Telco-Customer-Churn.csv`
- **Size:** ~7,000 rows, 21 features
- **Target:** `Churn` (binary — “Yes” or “No”)

### Key Variables
| **Type** | **Example Features**|
| ------------- | ------------- | 
| **Numerical**| `tenure`, `MonthlyCharges`, `TotalCharges` |
| **Categorical** | `Contract`, `PaymentMethod`, `InternetService`, `Partner`, `Dependents` |

## 🧠 Project Workflow
### 1️⃣ Exploratory Data Analysis & Preprocessing (`notebooks/01_eda_preprocessing.ipynb`)
- Loaded and explored raw data (`Telco-Customer-Churn.csv`).
- Handled missing or inconsistent values (`TotalCharges`).
- Converted categorical variables to consistent types.
- Encoded target variable: `Churn → 0/1`.
- Saved cleaned dataset → `data/df_preprocessed.csv`.

### 2️⃣ Modeling (`notebooks/02_modeling.ipynb`)
- Split data: 80% train / 20% test (stratified).
  - gg
- Built preprocessing pipeline:
- Models compared:
