import joblib
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from datetime import datetime

# Load pre-trained models
random_forest = joblib.load('randomforest.pkl')
knn_ = joblib.load('knn.pkl')
decision_tree = joblib.load('decisiontree.pkl')

# Replace these with real model accuracy values
model_accuracy1 = 99.79  # Random Forest accuracy in percentage
model_accuracy2 = 94.92  # KNN accuracy in percentage
model_accuracy3 = 100  # Decision Tree accuracy in percentage

# API key for fetching real-time gold prices
api_key = 'goldapi-1b9bbslywo2n1w-io'  # Replace with your GoldAPI key
api_url = 'https://www.goldapi.io/api/XAU/INR'
usd_to_inr_api_url = 'https://api.exchangerate-api.com/v4/latest/USD'

# Create Flask app
app = Flask(__name__)

# Define prediction function (dummy example, replace with actual feature generation)
def generate_features(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    # Example feature generation based on the date
    
    # Replace this with actual logic to generate features for your models
    features = np.array([date.year, date.month, date.day, date.weekday()]).reshape(1, -1)
    return features

# Function to fetch real-time gold price in INR
def get_real_time_gold_price():
    headers = {
        'x-access-token': api_key,
        'Content-Type': 'application/json'
    }
    response = requests.get(api_url, headers=headers)
    data = response.json()
    if 'price' in data:
        return data['price']
    else:
        return None

# Function to convert USD to INR
def convert_usd_to_inr(usd_value):
    response = requests.get(usd_to_inr_api_url)
    data = response.json()
    if 'rates' in data and 'INR' in data['rates']:
        return usd_value * data['rates']['INR']
    else:
        return None

# Function to convert troy ounce to pennyweight
def convert_ounce_to_pennyweight(ounce_price):
    return ounce_price / 20  # 1 pennyweight is 1/20th of a troy ounce

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form.get('date')
    features = generate_features(date_str)

    # Make predictions using each model
    prediction1_usd = random_forest.predict(features)[0]
    prediction2_usd = knn_.predict(features)[0]
    prediction3_usd = decision_tree.predict(features)[0]

    # Convert predicted values to INR
    prediction1_inr = convert_usd_to_inr(prediction1_usd)
    prediction2_inr = convert_usd_to_inr(prediction2_usd)
    prediction3_inr = convert_usd_to_inr(prediction3_usd)

    # Fetch real-time gold price in INR for 1 pennyweight
    actual_gold_price_per_ounce_inr = get_real_time_gold_price()
    actual_gold_price_per_pennyweight_inr = convert_ounce_to_pennyweight(actual_gold_price_per_ounce_inr)

    if actual_gold_price_per_pennyweight_inr is None:
        return jsonify({'error': 'Failed to fetch real-time data'}), 500

    # Calculate accuracy between predicted values and actual value
    accuracy1 = 100 - abs((actual_gold_price_per_pennyweight_inr - prediction1_inr) / actual_gold_price_per_pennyweight_inr) * 100
    accuracy2 = 100 - abs((actual_gold_price_per_pennyweight_inr - prediction2_inr) / actual_gold_price_per_pennyweight_inr) * 100
    accuracy3 = 100 - abs((actual_gold_price_per_pennyweight_inr - prediction3_inr) / actual_gold_price_per_pennyweight_inr) * 100

    # Return predictions as JSON
    return jsonify({
        'gold_price_model1_usd': prediction1_usd,
        'gold_price_model1_inr': prediction1_inr,
        'gold_price_model2_usd': prediction2_usd,
        'gold_price_model2_inr': prediction2_inr,
        'gold_price_model3_usd': prediction3_usd,
        'gold_price_model3_inr': prediction3_inr,
        'actual_value_per_pennyweight_inr': actual_gold_price_per_pennyweight_inr,
        'model_accuracy1': model_accuracy1,
        'model_accuracy2': model_accuracy2,
        'model_accuracy3': model_accuracy3,
        'accuracy1': accuracy1,
        'accuracy2': accuracy2,
        'accuracy3': accuracy3
    })

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
