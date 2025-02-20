import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Generate sample data with three features
np.random.seed(42)
data_size = 100
RM = 5 + np.random.rand(data_size, 1) * 3  # Avg rooms (5-8)
PTRATIO = 15 + np.random.rand(data_size, 1) * 10  # Pupil-teacher ratio (15-25)
LSTAT = np.random.rand(data_size, 1) * 40  # Lower status percentage (0-40)

# Target variable (Price)
y = 50000 + 10000 * RM - 2000 * PTRATIO - 1500 * LSTAT + np.random.randn(data_size, 1) * 5000

# Combine into a single dataset
X = np.hstack([RM, PTRATIO, LSTAT])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")

# Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
