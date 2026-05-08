import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

st.sidebar.header("Customer Information")

# Sidebar Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 840.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Create input dictionary
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Convert categorical manually (important)
if gender == "Male":
    input_data["gender_Male"] = 1

if partner == "Yes":
    input_data["Partner_Yes"] = 1

if internet == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet == "No":
    input_data["InternetService_No"] = 1

if contract == "One year":
    input_data["Contract_One year"] = 1
elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Match model columns
input_df = input_df.reindex(columns=model_columns, fill_value=0)

if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.write(f"Churn Probability: **{round(probability*100,2)}%**")
    st.subheader("📊 Feature Importance")


    import matplotlib.pyplot as plt

    # Create dataframe of feature importance
    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Coefficient": model.coef_[0]
    })

    # Sort by absolute importance
    importance_df["Abs_Coefficient"] = importance_df["Coefficient"].abs()
    importance_df = importance_df.sort_values(by="Abs_Coefficient", ascending=False).head(10)

    # Plot
    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Coefficient"])
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Top 10 Important Features")
    ax.invert_yaxis()

    st.pyplot(fig)