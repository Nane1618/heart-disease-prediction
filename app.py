from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model, scaler, and encoder
model = joblib.load("overall_best_model.pkl")
scaler = joblib.load("standard_encoder.dill")
ohe_encoder = joblib.load("ohe_encoder.dill")

# Patch for sklearn < 1.3 compatibility with newer sklearn versions
if not hasattr(ohe_encoder, 'feature_name_combiner'):
    ohe_encoder.feature_name_combiner = "concat"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/hospitals')
def hospitals():
    return render_template('index2.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    data = request.get_json()
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([data])

    # Ensure that all required columns exist, and set default values if missing
    for col in ['fbs', 'restecg']:
        if col not in input_data.columns:
            input_data[col] = 0  # Default value for missing columns

    # Define the categorical and numerical columns
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # One-Hot Encode categorical columns
    input_data_encoded = ohe_encoder.transform(input_data[categorical_columns])

    # Scale numerical columns
    input_data_scaled = scaler.transform(input_data[numerical_columns])

    # Combine encoded and scaled features into a single DataFrame
    input_data_final = pd.concat([
        pd.DataFrame(input_data_scaled, columns=numerical_columns),
        pd.DataFrame(input_data_encoded, columns=ohe_encoder.get_feature_names_out(categorical_columns))
    ], axis=1)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_final)

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Subscriber functionality
SUBSCRIBERS_FILE = 'subscribers.json'

def load_subscribers():
    """Load subscribers from JSON file"""
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_subscribers(subscribers):
    """Save subscribers to JSON file"""
    with open(SUBSCRIBERS_FILE, 'w') as f:
        json.dump(subscribers, f, indent=4)

@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Handle newsletter subscription"""
    data = request.get_json()
    email = data.get('email', '').strip()
    
    if not email:
        return jsonify({'success': False, 'message': 'Email is required'}), 400
    
    # Basic email validation
    if '@' not in email or '.' not in email:
        return jsonify({'success': False, 'message': 'Invalid email format'}), 400
    
    # Load existing subscribers
    subscribers = load_subscribers()
    
    # Check if already subscribed
    existing_emails = [s['email'] for s in subscribers]
    if email in existing_emails:
        return jsonify({'success': False, 'message': 'This email is already subscribed!'}), 400
    
    # Add new subscriber
    new_subscriber = {
        'email': email,
        'subscribed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    subscribers.append(new_subscriber)
    save_subscribers(subscribers)
    
    return jsonify({'success': True, 'message': 'Successfully subscribed! Thank you.'})

@app.route('/subscribers')
def view_subscribers():
    """View all subscribers (admin page)"""
    subscribers = load_subscribers()
    return render_template('subscribers.html', subscribers=subscribers)

if __name__ == "__main__":
    app.run(debug=True)
