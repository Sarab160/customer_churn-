# 📊 Customer Churn Prediction App (Streamlit + Random Forest)

This project is a Machine Learning based **Customer Churn Prediction Web App** built using:

- Python
- Scikit-Learn
- Random Forest Classifier
- Imbalanced-Learn (RandomOverSampler)
- Streamlit
- Pandas, Seaborn, Matplotlib

The app allows users to enter customer details and predict whether the customer is likely to churn or not.

---

## 🚀 Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave their service.

In this project:
- Data preprocessing is performed
- Missing values handled
- One-Hot Encoding applied
- Class imbalance handled using RandomOverSampler
- Random Forest model trained
- Model evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Classification Report

---

## 🧠 Machine Learning Pipeline

1. Load Dataset (`customer.csv`)
2. Clean & preprocess data
3. Encode categorical features
4. Handle class imbalance
5. Train-test split
6. Train Random Forest Classifier
7. Evaluate model performance
8. Deploy with Streamlit UI

---

## 📈 Model Performance

- Accuracy: ~90%
- Precision: ~86%
- Recall: ~96%
- F1 Score: ~91%

This shows strong predictive performance with high recall for churn detection.

---

username/customer-churn-app.git
cd customer-churn-app
