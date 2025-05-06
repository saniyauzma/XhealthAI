from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import mysql.connector
from datetime import datetime

# Define the blueprint
heart = Blueprint('heart', __name__)

# Load models and other resources
log_model = joblib.load('model/logistic_regression_model.pkl')
rf_model = joblib.load('model/random_forest_model.pkl')
abc = joblib.load('model/adaboost_model.pkl')
meta_learner = joblib.load('model/ensemble_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# MySQL database connection setup (connect when needed)
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="health_predictions"
    )

# Route to fetch user data and heart-related data
@heart.route('/heart_data/<int:user_id>', methods=['GET'])
def get_user_data(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        # Fetch user details from the user_data table
        cursor.execute("SELECT user_id, name, email, password, age, gender FROM user_data WHERE user_id = %s", (user_id,))
        user_data = cursor.fetchone()

        # Fetch the latest heart-related data from the heart_data table
        cursor.execute("""
            SELECT age, sex, chestPainType, restingBP, cholesterol, fastingBS, restingECG, maxHR, exerciseAngina, oldpeak, stSlope, restingBP
            FROM heart_data WHERE user_id = %s ORDER BY date_recorded DESC LIMIT 1
        """, (user_id,))
        heart_data = cursor.fetchone()

        if user_data and heart_data:
            user_details = {
                "user_id": user_data[0],  # user_id from user_data
                "name": user_data[1],     # Name from user_data
                "email": user_data[2],    # Email from user_data
                "password": user_data[3], # Password from user_data
                "age": user_data[4],      # Age from user_data
                "gender": user_data[5],   # Gender from user_data
                "heart_data": {
                    "age": heart_data[0],          # Age from heart_data
                    "sex": heart_data[1],          # Gender from heart_data
                    "chestPainType": heart_data[2],# Chest Pain Type
                    "restingBP": heart_data[3],    # Resting BP
                    "cholesterol": heart_data[4],  # Cholesterol
                    "fastingBS": heart_data[5],    # Fasting Blood Sugar
                    "restingECG": heart_data[6],   # Resting ECG
                    "maxHR": heart_data[7],        # Maximum Heart Rate
                    "exerciseAngina": heart_data[8],# Exercise Angina
                    "oldpeak": heart_data[9],      # Oldpeak
                    "stSlope": heart_data[10],     # ST Slope
                }
            }
            return jsonify({"user_details": user_details}), 200
        else:
            return jsonify({"error": "User data or heart data not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()

# Route to predict heart disease risk
@heart.route('/heart_predict', methods=['POST'])
def predict_heart_disease():
    data = request.get_json()
    print("Received data:", data)

    # Define mappings for categorical variables
    chest_pain_mapping = {
        "Typical": 0,
        "Atypical": 1,
        "Non-Anginal": 2,
        "Asymptomatic": 3
    }
    resting_ecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    st_slope_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    try:
        # Preprocess input data
        input_data = np.array([[
            int(data['age']),
            1 if data['sex'] == 'Male' else 0,
            chest_pain_mapping[data['chestPainType']],
            int(data['restingBP']),
            int(data['cholesterol']),
            int(data['fastingBS']),
            resting_ecg_mapping[data['restingECG']],
            int(data['maxHR']),
            1 if data['exerciseAngina'] == 'Yes' else 0,
            float(data['oldpeak']),
            st_slope_mapping[data['stSlope']]
        ]])

        
        log_pred = log_model.predict(input_data)
        rf_pred = rf_model.predict(input_data)
        abc_pred = abc.predict(input_data)

        
        X_stack = np.column_stack((log_pred, rf_pred, abc_pred))
        ensemble_prediction = meta_learner.predict(X_stack)
        ensemble_probs = meta_learner.predict_proba(X_stack)

        # Prepare response
        response = {
            'message': 'Heart disease risk detected!' if ensemble_prediction[0] == 1 else 'No heart disease risk detected.',
            'confidence': f"{round(ensemble_probs[0][1] * 100, 2)}%" if ensemble_prediction[0] == 1 else f"{round(ensemble_probs[0][0] * 100, 2)}%",
            'probabilities': {
                'no_risk': f"{round(ensemble_probs[0][0] * 100, 2)}%",
                'risk': f"{round(ensemble_probs[0][1] * 100, 2)}%"
            }
        }
        return jsonify(response), 200

    except KeyError as e:
        return jsonify({'error': f"Missing or invalid key: {str(e)}"}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to save heart data
@heart.route('/heart_data/<int:user_id>', methods=['POST'])
def save_heart_data(user_id):
    data = request.get_json()

    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO heart_data (
                user_id, age, sex, chestPainType, restingBP, cholesterol, fastingBS, restingECG, maxHR, exerciseAngina, oldpeak, stSlope, date_recorded
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            data['age'],
            data['sex'],
            data['chestPainType'],
            data['restingBP'],
            data['cholesterol'],
            data['fastingBS'],
            data['restingECG'],
            data['maxHR'],
            data['exerciseAngina'],
            data['oldpeak'],
            data['stSlope'],
            datetime.now()
        ))

        db_connection.commit()
        return jsonify({'message': 'Heart data saved successfully.'}), 201

    except KeyError as e:
        return jsonify({'error': f"Missing key: {str(e)}"}), 400

    except Exception as e:
        db_connection.rollback()
        return jsonify({'error': str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()






