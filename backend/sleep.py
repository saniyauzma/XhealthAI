from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import mysql.connector
from datetime import datetime

# Define the blueprint
sleep = Blueprint('sleep', __name__)

# Load models and other resources

rf_model = joblib.load(r'C:\Users\Harshitha\OneDrive\Desktop\mini\backend\models\best_random_grid.pkl')


# MySQL database connection setup (connect when needed)
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="health_predictions"
    )

# Route to fetch user data and sleep-related data
@sleep.route('/sleep_data/<int:user_id>', methods=['GET'])
def get_user_data(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        # Fetch user details from the user_data table
        cursor.execute("SELECT user_id, name, email, password, age, gender FROM user_data WHERE user_id = %s", (user_id,))
        user_data = cursor.fetchone()

        # Fetch the latest sleep-related data from the sleep_data table
        cursor.execute(""" 
            SELECT age, gender, sleep_duration, quality_of_sleep, physical_activity_level, stress_level, 
                   bmi_category, heart_rate, daily_steps, systolic_bp, diastolic_bp
            FROM sleep_data WHERE user_id = %s ORDER BY date_recorded DESC LIMIT 1
        """, (user_id,))
        sleep_data = cursor.fetchone()

        if user_data and sleep_data:
            user_details = {
                "user_id": user_data[0],  # user_id from user_data
                "name": user_data[1],     # Name from user_data
                "email": user_data[2],    # Email from user_data
                "password": user_data[3], # Password from user_data
                "age": user_data[4],      # Age from user_data
                "gender": user_data[5],   # Gender from user_data
                "sleep_data": {
                    "age": sleep_data[0],                # Age from sleep_data
                    "gender": sleep_data[1],             # Gender from sleep_data
                    "sleep_duration": sleep_data[2],     # Sleep Duration
                    "quality_of_sleep": sleep_data[3],   # Quality of Sleep
                    "physical_activity_level": sleep_data[4], # Physical Activity Level
                    "stress_level": sleep_data[5],        # Stress Level
                    "bmi_category": sleep_data[6],        # BMI Category
                    "heart_rate": sleep_data[7],          # Heart Rate
                    "daily_steps": sleep_data[8],         # Daily Steps
                    "systolic_bp": sleep_data[9],        # Systolic Blood Pressure
                    "diastolic_bp": sleep_data[10]       # Diastolic Blood Pressure
                }
            }
            return jsonify({"user_details": user_details}), 200
        else:
            return jsonify({"error": "User data or sleep data not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()

# Route to predict sleep disorder risk
@sleep.route('/sleep_predict', methods=['POST'])
def predict_sleep_disorder():
    data = request.get_json()
    print("Received data:", data)

    # Define mappings for categorical variables
    bmi_category_mapping = {
        "Underweight": 0,
        "Normal": 1,
        "Overweight": 2,
        "Obese": 3
    }

    try:
        # Preprocess input data
        input_data = np.array([[
            int(data['age']),
            1 if data['gender'] == 'Male' else 0,
            float(data['sleep_duration']),
            int(data['quality_of_sleep']),
            int(data['physical_activity_level']),
            int(data['stress_level']),
            bmi_category_mapping[data['bmi_category']],
            int(data['heart_rate']),
            int(data['daily_steps']),
            int(data['systolic_bp']),
            int(data['diastolic_bp'])
        ]])

        # Get predictions from individual models
        rf_prediction= rf_model.predict(input_data)
        rf_probs = rf_model.predict_proba(input_data)

        # Prepare response
        response = {
            'message': 'Sleep Apnea risk detected!' if rf_prediction[0] == 1 else 'No sleep disorder risk detected.',
            'confidence': f"{round(rf_probs[0][1] * 100, 2)}%" if rf_prediction[0] == 1 else f"{round(rf_probs[0][0] * 100, 2)}%",
            'probabilities': {
                'no_risk': f"{round(rf_probs[0][0] * 100, 2)}%",
                'risk': f"{round(rf_probs[0][1] * 100, 2)}%"
            }
        }
        return jsonify(response), 200

    except KeyError as e:
        return jsonify({'error': f"Missing or invalid key: {str(e)}"}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to save sleep data
@sleep.route('/sleep_data/<int:user_id>', methods=['POST'])
def save_sleep_data(user_id):
    data = request.get_json()

    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        cursor.execute(""" 
            INSERT INTO sleep_data (
                user_id, age, gender, sleep_duration, quality_of_sleep, physical_activity_level, stress_level, 
                bmi_category, heart_rate, daily_steps, systolic_bp, diastolic_bp, date_recorded
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            data['age'],
            data['gender'],
            data['sleep_duration'],
            data['quality_of_sleep'],
            data['physical_activity_level'],
            data['stress_level'],
            data['bmi_category'],
            data['heart_rate'],
            data['daily_steps'],
            data['systolic_bp'],
            data['diastolic_bp'],
            datetime.now()
        ))

        db_connection.commit()
        return jsonify({'message': 'Sleep data saved successfully.'}), 201

    except KeyError as e:
        return jsonify({'error': f"Missing key: {str(e)}"}), 400

    except Exception as e:
        db_connection.rollback()
        return jsonify({'error': str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()
