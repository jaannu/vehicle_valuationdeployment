import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from werkzeug.utils import secure_filename

# Flask App Initialization
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Machine Learning Models
valuation_model = joblib.load('model/valuation_model.pkl')  # CSV-based valuation model
condition_model = load_model('model/condition_model.h5')  # Image-based condition model

# Allowed file extensions
ALLOWED_CSV_EXTENSIONS = {'csv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper Functions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_input(data):
    # Example preprocessing steps:
    data.columns = [col.strip() for col in data.columns]  # Strip any extra spaces
    data = data[['RESIDUAL_QUOTE_AMOUNT', 'ASSET_COST', 'LOAN_AMOUNT','ACTUAL_LOAN_AMOUNT']]
    return data

# Valuation Calculation Function
def calculate_valuation(base_price, manufacture_year, mileage, fuel_type, brand, rto_number):
    # Constants
    CURRENT_YEAR = 2024
    AGE_PENALTY_RATE = 0.05  # 5% depreciation per year
    MILEAGE_PENALTY_RATE = 0.02  # 2% depreciation for every 10,000 km
    ELECTRIC_BONUS = 30000  # High bonus for electric vehicles
    DIESEL_PENALTY = 10000  # Penalty for diesel vehicles
    LUXURY_BRANDS = ['Mercedes', 'BMW', 'Audi', 'Toyota', 'Jaguar', 'Rolls Royce']  # Example luxury brands
    BRAND_BONUS = 0.05  # 5% extra for luxury brands

    # State-wise adjustment factors based on RTO number
    STATE_ADJUSTMENT = {
    'AP': 1.03,  # Andhra Pradesh: 3% increase
    'AR': 0.95,  # Arunachal Pradesh: 5% decrease
    'AS': 1.02,  # Assam: 2% increase
    'BR': 0.90,  # Bihar: 10% decrease
    'CG': 0.92,  # Chhattisgarh: 8% decrease
    'GA': 1.06,  # Goa: 6% increase
    'GJ': 0.90,  # Gujarat: 10% decrease
    'HR': 1.04,  # Haryana: 4% increase
    'HP': 1.01,  # Himachal Pradesh: 1% increase
    'JH': 0.92,  # Jharkhand: 8% decrease
    'KA': 0.95,  # Karnataka: 5% decrease
    'KL': 1.02,  # Kerala: 2% increase
    'MP': 0.94,  # Madhya Pradesh: 6% decrease
    'MH': 1.10,  # Maharashtra: 10% increase
    'MN': 0.93,  # Manipur: 7% decrease
    'ML': 0.94,  # Meghalaya: 6% decrease
    'MZ': 0.93,  # Mizoram: 7% decrease
    'NL': 0.94,  # Nagaland: 6% decrease
    'OR': 0.96,  # Odisha: 4% decrease
    'PB': 1.03,  # Punjab: 3% increase
    'RJ': 0.97,  # Rajasthan: 3% decrease
    'SK': 1.00,  # Sikkim: No adjustment
    'TN': 1.05,  # Tamil Nadu: 5% increase
    'TS': 1.03,  # Telangana: 3% increase
    'TR': 0.95,  # Tripura: 5% decrease
    'UP': 0.85,  # Uttar Pradesh: 15% decrease
    'UK': 0.98,  # Uttarakhand: 2% decrease
    'WB': 1.08,  # West Bengal: 8% increase
    'DL': 1.00,  # Delhi: No adjustment
    'JK': 0.92,  # Jammu & Kashmir: 8% decrease
    'LD': 1.00,  # Lakshadweep: No adjustment
    'PY': 1.02,  # Puducherry: 2% increase
    'CH': 1.01,  # Chandigarh: 1% increase
    'AN': 0.98,  # Andaman & Nicobar Islands: 2% decrease
    'DN': 0.95,  # Daman & Diu: 5% decrease
    'DD': 0.95,  # Dadra & Nagar Haveli: 5% decrease
    'LA': 0.93,  # Ladakh: 7% decrease
}

    # Extract state from RTO number
    state_code = rto_number[:2].upper()  # Assume first two characters are the state code

    # Determine state adjustment factor
    state_factor = STATE_ADJUSTMENT.get(state_code, 1.00)  # Default to no adjustment if state not found

    # Calculate age penalty (as a percentage of base price)
    age_years = CURRENT_YEAR - manufacture_year
    age_penalty = base_price * min(AGE_PENALTY_RATE * age_years, 0.6)  # Max penalty of 60%

    # Calculate mileage penalty (2% per 10,000 km)
    mileage_penalty = (mileage // 10000) * base_price * MILEAGE_PENALTY_RATE
    fuel_bonus = 0
    if fuel_type.lower() == "electric":
        fuel_bonus = ELECTRIC_BONUS
    brand_bonus = 0
    if brand in LUXURY_BRANDS:
        brand_bonus = base_price * BRAND_BONUS

    # Apply state adjustment
    state_adjustment = base_price * (state_factor - 1)

    # Calculate final valuation
    valuation = base_price - age_penalty - mileage_penalty + fuel_bonus + brand_bonus + state_adjustment

    # Ensure the valuation doesn't go below 0
    valuation = max(valuation, 0)

    return valuation


# Home Route
@app.route('/')
def index():
    return render_template('mainpage.html')

# Vehicle Valuation Prediction
@app.route('/predict_vehicle', methods=['POST'])
def predict_vehicle():
    try:
        # Collect form data
        model = request.form.get('model')
        brand = request.form.get('brand')
        rto_number = request.form.get('rto_number')
        mileage = int(request.form.get('mileage'))
        manufacture_year = int(request.form.get('manufacture_year'))
        fuel_type = request.form.get('fuel_type')
        base_price = int(request.form.get('base_price'))

        # Call the valuation calculation function
        valuation = calculate_valuation(base_price, manufacture_year, mileage, fuel_type, brand,rto_number)

        return jsonify({'valuation': f"₹{valuation:,.0f}"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# CSV Upload and Valuation Prediction
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read CSV and preprocess data
        try:
            data = pd.read_csv(filepath)
            preprocessed_data = preprocess_input(data)  # Pass the data to preprocess_input()
            predictions = valuation_model.predict(preprocessed_data)
            data['Predicted Valuation'] = predictions
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'valuations.csv')
            data.to_csv(output_filepath, index=False)
            return jsonify({'message': 'CSV processed successfully!', 'download_link': '/uploads/valuations.csv'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

# Serve Uploaded Files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Display Download Link or Confirmation Page
@app.route('/result')
def result():
    return render_template('result.html', download_link='/uploads/valuations.csv')

# Error Handling for Missing Files
@app.route('/error')
def error():
    return render_template('error.html', message="An error occurred while processing the file.")

# Image Upload and Condition Prediction
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image and predict condition
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            image = load_img(filepath, target_size=(224, 224))
            image_array = img_to_array(image) / 255.0
            image_array = image_array.reshape((1, 224, 224, 3))
            prediction = condition_model.predict(image_array)
            condition = ['Good', 'Average', 'Bad'][prediction.argmax()]
            return jsonify({'message': 'Image processed successfully!', 'condition': condition})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only image files (png, jpg, jpeg) are allowed.'}), 400

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
