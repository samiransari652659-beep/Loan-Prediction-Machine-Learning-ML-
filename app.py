
import streamlit as st
import pandas as pd
import joblib

# Load Model and Dataset
model=joblib.load('loan_model.pkl')
data = pd.read_csv('loan_data.csv')

st.title("Loan Approval Prediction")
st.write("Enter Applicant Details")

# Input Features

person_age = st.number_input("Person Age", 18, 100, 25)

person_gender = st.selectbox(
    "Gender", data['person_gender'].dropna().unique()
)

person_education = st.selectbox(
    "Education", data['person_education'].dropna().unique()
)

person_income = st.number_input("Person Income", value=50000)

person_emp_exp = st.number_input("Employment Experience", value=2)

person_home_ownership = st.selectbox(
    "Home Ownership", data['person_home_ownership'].dropna().unique()
)

loan_amnt = st.number_input("Loan Amount", value=10000)

loan_intent = st.selectbox(
    "Loan Intent", data['loan_intent'].dropna().unique()
)

loan_int_rate = st.number_input("Loan Interest Rate", value=10.0)

loan_percent_income = st.number_input(
    "Loan Percent Income", value=0.20
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length", value=3.0
)

credit_score = st.number_input("Credit Score", value=700)

previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Defaults",
    data['previous_loan_defaults_on_file'].dropna().unique()
)

# Prediction Button

if st.button("Predict Loan Status"):

    new_data = pd.DataFrame([{
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }])

    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.success("Loan Approved (YES)")
    else:
        st.error("Loan Rejected (NO)")