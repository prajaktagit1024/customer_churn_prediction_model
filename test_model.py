import pandas as pd
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training columns
with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Example new customer data (EDIT THIS TO TEST)
input_data = {
    "tenure": 12,
    "MonthlyCharges": 70,
    "TotalCharges": 840,
    "gender_Male": 1,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 1,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_Yes": 1,
    "DeviceProtection_Yes": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_Yes": 1,
    "StreamingMovies_Yes": 1,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "PaymentMethod_Credit card (automatic)": 0
}

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Ensure correct column order
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
prediction = model.predict(input_df)[0]

if prediction == 1:
    print("Customer is likely to CHURN ❌")
else:
    print("Customer is likely to STAY ✅")