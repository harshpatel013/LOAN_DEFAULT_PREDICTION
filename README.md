# Loan Default Prediction
### Machine Learning Classification Project

---

## Overview
This project predicts whether a bank loan applicant will default on their loan or not. It is a **binary classification problem** where the target variable is `Default` (1 = Defaulted, 0 = No Default). Three machine learning models are trained and compared to find the best predictor.

---

## Dataset
- **File:** `credit_risk_dataset.csv`
- **Rows:** 1000+ customer records
- **Target Column:** `Default` (0 = No Default, 1 = Default)

| Column | Description |
|---|---|
| Age | Age of the applicant |
| Income | Annual income |
| Loan_Amount | Loan amount requested |
| Credit_Score | Applicant's credit score |
| Employment_Years | Years of employment |
| Education_Level | Highest education level |
| Housing_Status | Own / Rent |
| Default | Target variable (0 or 1) |

---

## Project Structure
```
loan_default_prediction/
├── loan_default_prediction.py   # Main Python script
├── credit_risk_dataset.csv      # Dataset
├── eda_visualisations.png       # EDA plots
├── model_evaluation.png         # Accuracy, ROC, Feature Importance
├── confusion_matrices.png       # Confusion matrices for all 3 models
└── README.md
```

---

## Steps Followed

1. **Import Libraries** — pandas, numpy, matplotlib, seaborn, scikit-learn
2. **Load Dataset** — read CSV, check shape and preview rows
3. **Exploratory Data Analysis (EDA)** — class distribution, age histogram, credit score boxplot, scatter plot, education bar chart, correlation heatmap
4. **Preprocessing** — fill missing values with median, label encode categorical columns, create new feature `Debt_to_Income`
5. **Train/Test Split** — 80% train, 20% test (stratified)
6. **Feature Scaling** — StandardScaler applied for Logistic Regression
7. **Model Training** — Logistic Regression, Decision Tree, Random Forest
8. **Evaluation** — Accuracy, ROC-AUC, Confusion Matrix, Classification Report
9. **Visualisations** — accuracy comparison bar chart, ROC curves, feature importance
10. **Summary** — comparison table and key observations

---

## Models Used

| Model | Description |
|---|---|
| Logistic Regression | Simple linear model, good baseline |
| Decision Tree | Rule-based splitting, easy to interpret |
| Random Forest | Ensemble of trees, reduces overfitting |

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 86.00% | — |
| Decision Tree | 84.50% | — |
| Random Forest | 86.00% | — |

> **Best Model:** Logistic Regression / Random Forest (tied at 86%)

---

## Key Observations
- **Credit Score** is the most important feature for predicting loan default.
- The **Debt-to-Income ratio** (engineered feature) adds useful prediction signal.
- The dataset is **imbalanced** (~86% No Default), so ROC-AUC is a better metric than accuracy alone.
- Random Forest performs well due to averaging across multiple decision trees.

---

## How to Run

**1. Install dependencies:**
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

**2. Place both files in the same folder:**
```
credit_risk_dataset.csv
loan_default_prediction.py
```

**3. Run the script:**
```
python loan_default_prediction.py
```

**4. Output files will be generated in the same folder:**
- `eda_visualisations.png`
- `model_evaluation.png`
- `confusion_matrices.png`

---

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` & `seaborn` — data visualisation
- `scikit-learn` — machine learning models and evaluation

---

## Author
**HARSH KUMAR PATEL**
Machine Learning Course Project — 2025
