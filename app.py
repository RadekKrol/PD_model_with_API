import io
import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('model.pkl')
    calibrator = joblib.load('calibrator.pkl')
    encoder = joblib.load('encoder.pkl')
    model_features = model.feature_names_in_
    print("Models and feature names loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first.")
    model = None
except KeyError:
    print("Error: 'model' or 'feature_names' not found in model.pkl.")
    model = None


low_volume_categories = ['Books & Magazines', 'Candy', 'Car Electronics', 'Cleaning & Sanitary', 'Collectibles', 'Costumes & Party Supplies', 
                         'Diversified Entertainment', 'Diversified Health & Beauty Products', 'Erotic Clothing & Accessories', 'Marketplaces', 
                         'Music & Movies', 'Office Machines & Related Accessories (Excl. Computers)', 'Prints & Photos', 'Safety Products', 
                         'Tobacco', 'Travel Services', 'Video Games & Related Accessories']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            csv_data = io.StringIO(file.stream.read().decode("UTF8"))
            input_df = pd.read_csv(csv_data)

            input_df['loan_issue_date'] = pd.to_datetime(input_df['loan_issue_date'])

            card_expiry = pd.to_datetime(pd.DataFrame({'year': input_df['card_expiry_year'], 'month': input_df['card_expiry_month'], 'day': 1}))
            input_df['card_expiry_days'] = (card_expiry - input_df['loan_issue_date']).dt.days

            input_df['merchant_category_encoded'] = np.where(input_df['merchant_category'].isin(low_volume_categories), 'Other', input_df['merchant_category'])
            input_df['merchant_category_encoded'] = encoder.transform(input_df['merchant_category_encoded'])

            input_df['amount_repaid_14d_1m'] = (input_df['amount_repaid_14d'] / input_df['amount_repaid_1m']).fillna(0)
            input_df['amount_repaid_14d_3m'] = (input_df['amount_repaid_14d'] / input_df['amount_repaid_3m']).fillna(0)
            input_df['amount_repaid_14d_6m'] = (input_df['amount_repaid_14d'] / input_df['amount_repaid_6m']).fillna(0)
            input_df['amount_repaid_14d_1y'] = (input_df['amount_repaid_14d'] / input_df['amount_repaid_1y']).fillna(0)
            input_df['amount_repaid_3m_6m'] = (input_df['amount_repaid_3m'] / input_df['amount_repaid_6m']).fillna(0)
            input_df['amount_repaid_3m_1y'] = (input_df['amount_repaid_3m'] / input_df['amount_repaid_1y']).fillna(0)
            input_df['amount_repaid_6m_1y'] = (input_df['amount_repaid_6m'] / input_df['amount_repaid_1y']).fillna(0)
            input_df['num_failed_payments_3m_1y'] = (input_df['num_failed_payments_3m'] / input_df['num_failed_payments_1y']).fillna(0)


            input_df['Intangible products'] = np.where(input_df['merchant_group'] == 'Intangible products', 1, 0)
            input_df['Food & Beverage'] = np.where(input_df['merchant_group'] == 'Food & Beverage', 1, 0)
            input_df['Jewelry & Accessories'] = np.where(input_df['merchant_group'] == 'Jewelry & Accessories', 1, 0)

            if not all(feature in input_df.columns for feature in model_features):
                missing_cols = list(set(model_features) - set(input_df.columns))
                return jsonify({'error': f'Missing columns in CSV: {missing_cols}'}), 400

            input_df = input_df[model_features]

        except Exception as e:
            return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 400

        try:
            predictions = model.predict_proba(input_df).T[1]
            probabilities = calibrator.predict(predictions)
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

        input_df['probability_of_1'] = probabilities
        response_json = input_df['probability_of_1'].to_json(orient='records')

        return Response(response_json, mimetype='application/json')
    return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/', methods=['GET'])
def health_check():
    if model:
        return "API is running and model is loaded."
    else:
        return "API is running, but the model failed to load.", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)