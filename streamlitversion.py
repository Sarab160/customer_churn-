import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

st.title("Customer Churn Prediction App")

# =============================
# Load Dataset
# =============================
df = pd.read_csv("customer.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df["TotalCharges"].mean(), inplace=True)

# =============================
# Feature Selection
# =============================
X_num = df[["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]]
Y = df["Churn"]

feature = df[["gender","Partner","Dependents","PhoneService","MultipleLines",
              "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
              "TechSupport","StreamingTV","StreamingMovies","Contract",
              "PaperlessBilling","PaymentMethod"]]

# One Hot Encoding
ohe = OneHotEncoder(sparse_output=False, drop="first")
encode_array = ohe.fit_transform(feature)
get_columns = ohe.get_feature_names_out(feature.columns)
encode_data = pd.DataFrame(encode_array, columns=get_columns)

Y = pd.get_dummies(Y, drop_first=True)

x_final = pd.concat([X_num, encode_data], axis=1)

# =============================
# Oversampling
# =============================
ru = RandomOverSampler()
x_ru, y_ru = ru.fit_resample(x_final, Y)

# =============================
# Train Test Split
# =============================
x_train, x_test, y_train, y_test = train_test_split(
    x_ru, y_ru, test_size=0.2, random_state=42
)

# =============================
# Model Training
# =============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# =============================
# Model Evaluation
# =============================
y_pred = model.predict(x_test)

accuracy = model.score(x_test, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cf = confusion_matrix(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

fig, ax = plt.subplots()
sns.heatmap(cf, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# =============================
# User Input Section
# =============================
st.subheader("Enter Customer Details")

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

gender = st.selectbox("Gender", df["gender"].unique())
Partner = st.selectbox("Partner", df["Partner"].unique())
Dependents = st.selectbox("Dependents", df["Dependents"].unique())
PhoneService = st.selectbox("Phone Service", df["PhoneService"].unique())
MultipleLines = st.selectbox("Multiple Lines", df["MultipleLines"].unique())
InternetService = st.selectbox("Internet Service", df["InternetService"].unique())
OnlineSecurity = st.selectbox("Online Security", df["OnlineSecurity"].unique())
OnlineBackup = st.selectbox("Online Backup", df["OnlineBackup"].unique())
DeviceProtection = st.selectbox("Device Protection", df["DeviceProtection"].unique())
TechSupport = st.selectbox("Tech Support", df["TechSupport"].unique())
StreamingTV = st.selectbox("Streaming TV", df["StreamingTV"].unique())
StreamingMovies = st.selectbox("Streaming Movies", df["StreamingMovies"].unique())
Contract = st.selectbox("Contract", df["Contract"].unique())
PaperlessBilling = st.selectbox("Paperless Billing", df["PaperlessBilling"].unique())
PaymentMethod = st.selectbox("Payment Method", df["PaymentMethod"].unique())

# =============================
# Prediction Button
# =============================
if st.button("Predict Churn"):

    input_data = pd.DataFrame([[SeniorCitizen, tenure, MonthlyCharges, TotalCharges]],
                              columns=X_num.columns)

    cat_data = pd.DataFrame([[gender, Partner, Dependents, PhoneService,
                              MultipleLines, InternetService, OnlineSecurity,
                              OnlineBackup, DeviceProtection, TechSupport,
                              StreamingTV, StreamingMovies, Contract,
                              PaperlessBilling, PaymentMethod]],
                            columns=feature.columns)

    cat_encoded = ohe.transform(cat_data)
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=get_columns)

    final_input = pd.concat([input_data, cat_encoded_df], axis=1)

    prediction = model.predict(final_input)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn ❌")
    else:
        st.success("Customer is NOT likely to Churn ✅")