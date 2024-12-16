
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
    # 1. Rename columns to match model's expected column names
    data.columns = [col.strip() for col in data.columns]  # Strip any extra spaces
    data['vehicle_type'] = data['vehicle_type'].astype('category').cat.codes
    data = data[['manufacture_year','mileage', 'vehicle_type']] 
    # 2. Apply any necessary transformations or scaling
    # Example: data['Column1'] = scaler.transform(data['Column1'])
    return data

# Home Route
@app.route('/')
def index():
    return render_template('mainpage.html')

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
