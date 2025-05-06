from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import mysql.connector
from datetime import datetime

# Define the blueprint
pcos = Blueprint('pcos', __name__)

xg = joblib.load('modelp/xgboost_model.pkl')
ada = joblib.load('modelp/adaboost_model.pkl')
grad = joblib.load('modelp/gradientboost_model.pkl')
meta_learner = joblib.load('modelp/ensemble_model.pkl')

# MySQL database connection setup (connect when needed)
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root123",
        database="health_predictions"
    )

# Route to fetch user data and PCOS-related data
@pcos.route('/pcos_data/<int:user_id>', methods=['GET'])
def get_user_data(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        # Fetch user details from the user_data table
        cursor.execute("SELECT user_id, name, email, password, age, gender FROM user_data WHERE user_id = %s", (user_id,))
        user_data = cursor.fetchone()

        # Fetch the latest PCOS-related data from the pcos_data table
        cursor.execute("""
            SELECT age, weight, height, bmi, blood_group, pulse_rate, rr, hb, cycle_type, cycle_length, marriage_status, pregnant, 
                   num_of_abortions, beta_hcg_1, beta_hcg_2, fsh, lh, fsh_lh, hip, waist, waist_hip_ratio, tsh, amh, prl, vit_d3, 
                   prg, rbs, weight_gain, hair_growth, skin_darkening, hair_loss, pimples, fast_food, reg_exercise, bp_systolic, 
                   bp_diastolic, follicle_no_left, follicle_no_right, avg_f_size_left, avg_f_size_right, endometrium
            FROM pcos_data WHERE user_id = %s ORDER BY date_recorded DESC LIMIT 1
        """, (user_id,))
        pcos_data = cursor.fetchone()

        if user_data and pcos_data:
            user_details = {
                "user_id": user_data[0],  # user_id from user_data
                "name": user_data[1],     # Name from user_data
                "email": user_data[2],    # Email from user_data
                "password": user_data[3], # Password from user_data
                "age": user_data[4],      # Age from user_data
                "gender": user_data[5],   # Gender from user_data
                "pcos_data": {
                    "age": pcos_data[0],  # Age from pcos_data
                    "weight": pcos_data[1],  # Weight
                    "height": pcos_data[2],  # Height
                    "bmi": pcos_data[3],  # BMI
                    "blood_group": pcos_data[4],  # Blood Group
                    "pulse_rate": pcos_data[5],  # Pulse Rate
                    "rr": pcos_data[6],  # RR
                    "hb": pcos_data[7],  # Hb
                    "cycle_type": pcos_data[8],  # Cycle Type
                    "cycle_length": pcos_data[9],  # Cycle Length
                    "marriage_status": pcos_data[10],  # Marriage Status
                    "pregnant": pcos_data[11],  # Pregnant
                    "num_of_abortions": pcos_data[12],  # No. of Abortions
                    "beta_hcg_1": pcos_data[13],  # Beta HCG 1
                    "beta_hcg_2": pcos_data[14],  # Beta HCG 2
                    "fsh": pcos_data[15],  # FSH
                    "lh": pcos_data[16],  # LH
                    "fsh_lh": pcos_data[17],  # FSH/LH ratio
                    "hip": pcos_data[18],  # Hip Measurement
                    "waist": pcos_data[19],  # Waist Measurement
                    "waist_hip_ratio": pcos_data[20],  # Waist-Hip Ratio
                    "tsh": pcos_data[21],  # TSH
                    "amh": pcos_data[22],  # AMH
                    "prl": pcos_data[23],  # PRL
                    "vit_d3": pcos_data[24],  # Vitamin D3
                    "prg": pcos_data[25],  # Progesterone
                    "rbs": pcos_data[26],  # RBS
                    "weight_gain": pcos_data[27],  # Weight Gain
                    "hair_growth": pcos_data[28],  # Hair Growth
                    "skin_darkening": pcos_data[29],  # Skin Darkening
                    "hair_loss": pcos_data[30],  # Hair Loss
                    "pimples": pcos_data[31],  # Pimples
                    "fast_food": pcos_data[32],  # Fast Food Consumption
                    "reg_exercise": pcos_data[33],  # Regular Exercise
                    "bp_systolic": pcos_data[34],  # Systolic BP
                    "bp_diastolic": pcos_data[35],  # Diastolic BP
                    "follicle_no_left": pcos_data[36],  # Follicle Number Left
                    "follicle_no_right": pcos_data[37],  # Follicle Number Right
                    "avg_f_size_left": pcos_data[38],  # Avg Follicle Size Left
                    "avg_f_size_right": pcos_data[39],  # Avg Follicle Size Right
                    "endometrium": pcos_data[40],  # Endometrium
                }
            }
            return jsonify({"user_details": user_details}), 200
        else:
            return jsonify({"error": "User data or PCOS data not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()

# Route to predict PCOS risk
@pcos.route('/pcos_predict', methods=['POST'])
def predict_pcos():
    data = request.get_json()
    print("Received data:", data)

    try:
        # Preprocess input data (this example assumes you have feature mappings, similar to the heart prediction)
        input_data = np.array([[  # Assume data is preprocessed here
            float(data['age']),
            float(data['weight']),
            float(data['height']),
            float(data['bmi']),
            int(data['blood_group']),
            int(data['pulse_rate']),
            int(data['rr']),
            float(data['hb']),
            int(data['cycle_type']),
            int(data['cycle_length']),
            int(data['marriage_status']),
            int(data['pregnant']),
            int(data['num_of_abortions']),
            float(data['beta_hcg_1']),
            data['beta_hcg_2'],  # Assuming this is encoded or treated in some way
            float(data['fsh']),
            float(data['lh']),
            float(data['fsh_lh']),
            float(data['hip']),
            float(data['waist']),
            float(data['waist_hip_ratio']),
            float(data['tsh']),
            float(data['amh']),
            float(data['prl']),
            float(data['vit_d3']),
            float(data['prg']),
            float(data['rbs']),
            int(data['weight_gain']),
            int(data['hair_growth']),
            int(data['skin_darkening']),
            int(data['hair_loss']),
            int(data['pimples']),
            float(data['fast_food']),
            int(data['reg_exercise']),
            int(data['bp_systolic']),
            int(data['bp_diastolic']),
            int(data['follicle_no_left']),
            int(data['follicle_no_right']),
            float(data['avg_f_size_left']),
            float(data['avg_f_size_right']),
            float(data['endometrium']),
        ]])

        # Get predictions from individual models
        ada_pred = ada.predict(input_data)
        grad_pred = grad.predict(input_data)
        xg_pred = xg.predict(input_data)

        # Combine model predictions and make an ensemble prediction
        X_stack = np.column_stack((ada_pred, grad_pred, xg_pred))
        ensemble_prediction = meta_learner.predict(X_stack)
        ensemble_probs = meta_learner.predict_proba(X_stack)

        # Prepare response
        response = {
            'message': 'PCOS risk detected!' if ensemble_prediction[0] == 1 else 'No PCOS risk detected.',
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

# Route to save PCOS data
@pcos.route('/pcos_data/<int:user_id>', methods=['POST'])
def save_pcos_data(user_id):
    data = request.get_json()

    db_connection = get_db_connection()
    cursor = db_connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO pcos_data (
                user_id, age, weight, height, bmi, blood_group, pulse_rate, rr, hb, cycle_type, cycle_length, marriage_status, pregnant, 
                num_of_abortions, beta_hcg_1, beta_hcg_2, fsh, lh, fsh_lh, hip, waist, waist_hip_ratio, tsh, amh, prl, vit_d3, 
                prg, rbs, weight_gain, hair_growth, skin_darkening, hair_loss, pimples, fast_food, reg_exercise, bp_systolic, 
                bp_diastolic, follicle_no_left, follicle_no_right, avg_f_size_left, avg_f_size_right, endometrium, date_recorded
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            data['age'],
            data['weight'],
            data['height'],
            data['bmi'],
            data['blood_group'],
            data['pulse_rate'],
            data['rr'],
            data['hb'],
            data['cycle_type'],
            data['cycle_length'],
            data['marriage_status'],
            data['pregnant'],
            data['num_of_abortions'],
            data['beta_hcg_1'],
            data['beta_hcg_2'],
            data['fsh'],
            data['lh'],
            data['fsh_lh'],
            data['hip'],
            data['waist'],
            data['waist_hip_ratio'],
            data['tsh'],
            data['amh'],
            data['prl'],
            data['vit_d3'],
            data['prg'],
            data['rbs'],
            data['weight_gain'],
            data['hair_growth'],
            data['skin_darkening'],
            data['hair_loss'],
            data['pimples'],
            data['fast_food'],
            data['reg_exercise'],
            data['bp_systolic'],
            data['bp_diastolic'],
            data['follicle_no_left'],
            data['follicle_no_right'],
            data['avg_f_size_left'],
            data['avg_f_size_right'],
            data['endometrium'],
            datetime.now()
        ))

        db_connection.commit()

        return jsonify({"message": "PCOS data saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()
