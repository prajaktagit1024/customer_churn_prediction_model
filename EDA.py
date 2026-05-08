import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
df = pd.read_csv("data/Telco-Customer-Churn.csv")
print("shape of dataset:" , df.shape)
print(df.head())
df.info()
df.describe()
print(df.isnull().sum())

plt.figure()
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

print(df['Churn'].value_counts(normalize=True)*100)

plt.figure()
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Gender vs Churn")
plt.show()