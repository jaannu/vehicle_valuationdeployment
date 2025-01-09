import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load and preprocess data
try:
    # Load CSV with low_memory=False
    data = pd.read_csv('sold_data_tw_v2.csv', low_memory=False)
    
    # Identify relevant columns
    relevant_columns = ['RESIDUAL_QUOTE_AMOUNT', 'ASSET_COST', 'LOAN_AMOUNT','ACTUAL_LOAN_AMOUNT','NET_LOSS']
    data = data[relevant_columns]  # Keep only necessary columns

    # Convert relevant columns to numeric, handling errors
    for col in relevant_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with NaN values in relevant columns
    data = data.dropna()

except FileNotFoundError:
    print("Error: The file 'sold_data_tw_v2.csv' was not found.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Define features and target
X = data[['RESIDUAL_QUOTE_AMOUNT', 'ASSET_COST', 'LOAN_AMOUNT','ACTUAL_LOAN_AMOUNT']]
y = data['NET_LOSS']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
try:
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error training the model: {e}")
    exit()

# Evaluate
try:
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)  # Compute MSE
    rmse = np.sqrt(mse)  # Take square root of MSE
    print(f"RMSE: {rmse}")
except Exception as e:
    print(f"Error evaluating the model: {e}")
    exit()

# Save model
try:
    joblib.dump(model, 'model/valuation_model.pkl')
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")
    exit()
