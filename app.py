import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("loan_approvalt.pkl")
scaler = joblib.load("scaler_loan.pkl")


st.title("🏦 Loan Approval Predictor")

# Sidebar input
st.sidebar.header("Enter Details")

dependents = st.sidebar.slider("Dependents", 0, 10, 1)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income = st.sidebar.number_input("Income")
loan_amount = st.sidebar.number_input("Loan Amount")
loan_term = st.sidebar.number_input("Loan Term")
cibil = st.sidebar.slider("CIBIL Score", 300, 900, 700)

res_assets = st.sidebar.number_input("Residential Assets")
com_assets = st.sidebar.number_input("Commercial Assets")
lux_assets = st.sidebar.number_input("Luxury Assets")
bank_assets = st.sidebar.number_input("Bank Assets")

# Encoding
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if st.button("Predict"):
    data = np.array([[dependents, education, self_employed, income,
                      loan_amount, loan_term, cibil,
                      res_assets, com_assets, lux_assets, bank_assets]])

    data = scaler.transform(data)
    result = model.predict(data)
    prob = model.predict_proba(data)

    if result[0] == 1:
        st.success(f"✅ Approved (Confidence: {prob[0][1]*100:.2f}%)")
    else:
        st.error(f"❌ Rejected (Confidence: {prob[0][0]*100:.2f}%)")










# st.title("🏦 Loan Approval Prediction App")

# # Inputs
# dependents = st.number_input("No of Dependents", 0, 10)
# education = st.selectbox("Education", ["Graduate", "Not Graduate"])
# self_employed = st.selectbox("Self Employed", ["Yes", "No"])
# income = st.number_input("Annual Income")
# loan_amount = st.number_input("Loan Amount")
# loan_term = st.number_input("Loan Term")
# cibil = st.number_input("CIBIL Score")
# res_assets = st.number_input("Residential Assets Value")
# com_assets = st.number_input("Commercial Assets Value")
# lux_assets = st.number_input("Luxury Assets Value")
# bank_assets = st.number_input("Bank Asset Value")

# # Encoding
# education = 1 if education == "Graduate" else 0
# self_employed = 1 if self_employed == "Yes" else 0

# # Prediction
# if st.button("Predict"):
#     data = np.array([[dependents, education, self_employed, income,
#                       loan_amount, loan_term, cibil,
#                       res_assets, com_assets, lux_assets, bank_assets]])
    
#     data = scaler.transform(data)
    
#     prediction = model.predict(data)
    
#     if prediction[0] == 1:
#         st.success("✅ Loan Approved")
#     else:
#         st.error("❌ Loan Rejected")

  

