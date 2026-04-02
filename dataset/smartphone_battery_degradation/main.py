import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
df = pd.read_csv("smartphone_battery_degradation_data.csv")

# Print info
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Plot heatmap (optional)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Distribution plot (optional)
sns.histplot(df["battery_health_percent"], kde=True)
plt.title("Battery Health Distribution")
plt.show()

# Prepare features and target
X = df.drop(columns='battery_health_percent')
y = df['battery_health_percent']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = model.predict(X_test)
print("R² Score:", r2_score(Y_test, Y_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred)))

# Plot Actual vs Predicted (optional)
plt.figure(figsize=(8,5))
plt.scatter(Y_test, Y_pred, color='green')
plt.xlabel("Actual Battery Health")
plt.ylabel("Predicted Battery Health")
plt.title("Actual vs Predicted Battery Health")
plt.plot([50, 100], [50, 100], 'r--')
plt.show()

# Save model
joblib.dump(model, 'battery_model.pkl')
print("Model saved as battery_model.pkl")
