from flask import Blueprint, jsonify
import mysql.connector

# Define the blueprint for dashboard
dashboard = Blueprint('dashboard', __name__)

# MySQL database connection setup
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="health_predictions"
    )

# Route to fetch heart rate and cholesterol data for a specific user
@dashboard.route('/heart_data/<int:user_id>', methods=['GET'])
def get_heart_data(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        # Fetch heart rate and cholesterol data, sorted by date_recorded
        cursor.execute("""
            SELECT record_id, maxHR, cholesterol, date_recorded 
            FROM heart_data 
            WHERE user_id = %s 
            ORDER BY date_recorded DESC
            LIMIT 6           
        """, (user_id,))
        heart_data = cursor.fetchall()

        if heart_data:
            heart_rate_data = {
                "labels": [str(record[3]) for record in heart_data],  # Date Recorded
                "heart_rates": [record[1] for record in heart_data],  # maxHR
                "cholesterol": [record[2] for record in heart_data],  # Cholesterol values
                "threshold": 100  # Heart rate threshold
            }
            return jsonify({"heart_rate_data": heart_rate_data}), 200
        else:
            return jsonify({"error": "No heart data found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()

# Route to fetch sleep data for a specific user
@dashboard.route('/sleep_data/<int:user_id>', methods=['GET'])
def get_sleep_data(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        # Fetch sleep data, sorted by date_recorded
        cursor.execute("""
            SELECT record_id, sleep_duration, quality_of_sleep, stress_level, 
                   physical_activity_level, bmi_category, systolic_bp, diastolic_bp, daily_steps, date_recorded
            FROM sleep_data 
            WHERE user_id = %s 
            ORDER BY date_recorded  DESC
            LIMIT 6                
        """, (user_id,))
        sleep_data = cursor.fetchall()
        print(sleep_data) 

        if sleep_data:
            sleep_metrics = {
                "labels": [str(record[9]) for record in sleep_data],  # Date Recorded
                "sleep_duration": [record[1] for record in sleep_data],  # Sleep Duration
                "quality_of_sleep": [record[2] for record in sleep_data],  # Quality of Sleep
                "stress_level": [record[3] for record in sleep_data],  # Stress Level
                "physical_activity": [record[4] for record in sleep_data],  # Physical Activity Level
                "bmi_category": [record[5] for record in sleep_data],
                "systolic_bp": [record[6] for record in sleep_data],
                "diastolic_bp": [record[7] for record in sleep_data],
                "daily_steps": [record[8] for record in sleep_data]  # Added daily steps
            }
            return jsonify({"sleep_data": sleep_metrics}), 200
        else:
            return jsonify({"error": "No sleep data found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()

@dashboard.route('/diabetes_data/<int:user_id>', methods=['GET'])
def get_diabetes_data(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        # Fetch hypertension, HbA1c level, and blood glucose level data from diabetes_data
        cursor.execute("""
            SELECT hypertension, HbA1c_level, blood_glucose_level, date_recorded
            FROM diabetes_data 
            WHERE user_id = %s 
            ORDER BY date_recorded ASC
        """, (user_id,))
        diabetes_data = cursor.fetchall()

        if diabetes_data:
            diabetes_metrics = {
                "labels": [str(record[3]) for record in diabetes_data],  # Date Recorded
                "hypertension": [record[0] for record in diabetes_data],  # Hypertension
                "HbA1c_level": [record[1] for record in diabetes_data],  # HbA1c Level
                "blood_glucose_level": [record[2] for record in diabetes_data]  # Blood Glucose Level
            }
            return jsonify({"diabetes_data": diabetes_metrics}), 200
        else:
            return jsonify({"error": "No diabetes data found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()
