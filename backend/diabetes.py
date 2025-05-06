import joblib
import numpy as np
import mysql.connector
from flask import Blueprint, jsonify, request
from datetime import datetime

# Define the blueprint
diabetes = Blueprint('diabetes', __name__)

# Load models and other resources
grad_model = joblib.load('modeld/gradient_booster.pkl')

# MySQL database connection setup (connect when needed)
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="health_predictions"
    )

# Route to fetch user data and diabetes-related data
@diabetes.route('/diabetes_data/<int:user_id>', methods=['GET'])
def get_user_data(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        # Fetch user details from the user_data table
        cursor.execute("SELECT user_id, name, email, password, age, gender FROM user_data WHERE user_id = %s", (user_id,))
        user_data = cursor.fetchone()

        # Fetch the latest diabetes-related data from the diabetes_data table
        cursor.execute("""
            SELECT age, gender, hypertension, smoking_history, bmi, HbA1c_level, blood_glucose_level
            FROM diabetes_data WHERE user_id = %s ORDER BY date_recorded DESC LIMIT 1
        """, (user_id,))
        diabetes_data = cursor.fetchone()

        if user_data and diabetes_data:
            user_details = {
                "user_id": user_data[0],
                "name": user_data[1],
                "email": user_data[2],
                "password": user_data[3],
                "age": user_data[4],
                "gender": user_data[5],
                "diabetes_data": {
                    "age": diabetes_data[0],
                    "gender": diabetes_data[1],
                    "hypertension": diabetes_data[2],
                    "smoking_history": diabetes_data[3],
                    "bmi": diabetes_data[4],
                    "HbA1c_level": diabetes_data[5],
                    "blood_glucose_level": diabetes_data[6]
                }
            }
            return jsonify({"user_details": user_details}), 200
        else:
            return jsonify({"error": "User data or diabetes data not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()

# Route to predict diabetes risk
@diabetes.route('/diabetes_predict', methods=['POST'])
def predict_diabetes():
    data = request.get_json()
    print("Received data:", data)

    # Define mappings for categorical variables
    smoking_history_mapping = {
        "never": 0,
        "current": 1,
        "former": 2,
        "No Info": 3
    }

    try:
        # Preprocess input data
        input_data = np.array([[
            1 if data['gender'] == 'Female' else 0,
            int(data['age']),
            int(data['hypertension']),
            smoking_history_mapping[data['smoking_history']],
            float(data['bmi']),
            float(data['HbA1c_level']),
            float(data['blood_glucose_level'])
        ]])

        # Get predictions from individual models
        grad_pred = grad_model.predict(input_data)
        grad_preds = grad_model.predict_proba(input_data)

        response = {
            'message': 'Diabetes risk detected!' if grad_pred[0] == 1 else 'No diabetes risk detected.',
            'probabilities': {
                'no_risk': f"{round(grad_preds[0][0] * 100, 2)}%",
                'risk': f"{round(grad_preds[0][1] * 100, 2)}%"
            }
        }
        return jsonify(response), 200

    except KeyError as e:
        return jsonify({'error': f"Missing or invalid key: {str(e)}"}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to save diabetes data
@diabetes.route('/diabetes_data/<int:user_id>', methods=['POST'])
def save_diabetes_data(user_id):
    data = request.get_json()

    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO diabetes_data (
                user_id, gender, age, hypertension, smoking_history, bmi, HbA1c_level, blood_glucose_level
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            data['gender'],
            data['age'],
            data['hypertension'],
            data['smoking_history'],
            data['bmi'],
            data['HbA1c_level'],
            data['blood_glucose_level']
        ))

        db_connection.commit()
        return jsonify({'message': 'Diabetes data saved successfully.'}), 201

    except KeyError as e:
        return jsonify({'error': f"Missing key: {str(e)}"}), 400

    except Exception as e:
        db_connection.rollback()
        return jsonify({'error': str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()
