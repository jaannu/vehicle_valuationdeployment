import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
# Load and preprocess data
data = pd.read_csv('vehicle_data.csv')
data['vehicle_type'] = data['vehicle_type'].astype('category').cat.codes  # Encode categorical data
X = data[['manufacture_year', 'mileage', 'vehicle_type']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)  # Compute MSE
rmse = np.sqrt(mse)  # Take square root of MSE

print(f"RMSE: {rmse}")

# Save model
joblib.dump(model, 'model/valuation_model.pkl')
