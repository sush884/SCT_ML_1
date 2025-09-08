# model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Select relevant features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X_train = train_data[features]
y_train = train_data[target]

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on training data for evaluation
train_preds = model.predict(X_train)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_train, train_preds))
r2 = r2_score(y_train, train_preds)

print(f"Training RMSE: {rmse:.2f}")
print(f"Training R² Score: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_train, train_preds, alpha=0.5, color='teal')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig("price_prediction_plot.png")
plt.show()

# Predict on test data
X_test = test_data[features]
test_preds = model.predict(X_test)

# Save predictions to CSV
output = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_preds
})
output.to_csv("predictions.csv", index=False)

print("\n✅ Predictions saved to predictions.csv")
print("📊 Plot saved to price_prediction_plot.png")
