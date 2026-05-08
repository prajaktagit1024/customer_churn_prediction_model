import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Drop unnecessary column
df.drop("customerID", axis=1, inplace=True)

# Replace blank spaces with NaN
df = df.replace(" ", np.nan)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill all numeric NaN with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert target column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

# Apply one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Final safety check
print("Total NaN values:", df.isna().sum().sum())

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Extra safety (very important)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save column order
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model and columns saved successfully!")