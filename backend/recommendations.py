from flask import Blueprint, jsonify
import mysql.connector

# Define the blueprint for recommendations
recommendations = Blueprint('recommendations', __name__)

# MySQL database connection setup
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Update with your database user
        password="root123",  # Update with your database password
        database="health_predictions"  # Update with your database name
    )
 # Function to generate recommendations based on user data
def generate_recommendations(data):
    recommendations = []

    # Extract data with checks for None values
    cholesterol = data.get('cholesterol')
    restingBP = data.get('restingBP')
    fastingBS = data.get('fastingBS')
    exerciseAngina = data.get('exerciseAngina')
    maxHR = data.get('maxHR')
    restingECG = data.get('restingECG')
    sleep_duration = data.get('sleep_duration')
    stress_level = data.get('stress_level')
    heart_rate = data.get('heart_rate')
    bmi_category = data.get('bmi_category')
    daily_steps = data.get('daily_steps')
    systolic_bp = data.get('systolic_bp')
    diastolic_bp = data.get('diastolic_bp')

    # Cholesterol Recommendations
    if cholesterol is not None:
        if cholesterol > 240:
            recommendations.append("Your cholesterol level is high. This increases the risk of cardiovascular diseases. To lower cholesterol, consider incorporating more fiber into your diet, reducing saturated fats, and increasing physical activity through regular cardiovascular exercise.")
        elif cholesterol < 120:
            recommendations.append("Your cholesterol level is quite low. While this might be ideal for some, extremely low cholesterol can also be linked to other health risks. Ensure you have a balanced diet and consult your doctor about maintaining healthy cholesterol levels.")
        else:
            recommendations.append("Your cholesterol level is in the normal range. Maintain a balanced diet and exercise regularly to keep it within this healthy range.")

    # Resting BP Recommendations
    if restingBP is not None:
        if restingBP > 140:
            recommendations.append("Your resting blood pressure is high. High blood pressure can lead to heart disease and stroke. Reduce sodium intake, increase your physical activity, and consider medication under the guidance of a healthcare provider.")
        elif restingBP < 90:
            recommendations.append("Your resting blood pressure is low. While low blood pressure can be normal for some, if you experience dizziness or fainting, increase your salt intake and consult a healthcare provider.")
        else:
            recommendations.append("Your resting blood pressure is within a normal range. Continue with a balanced diet, regular exercise, and stress management to maintain it.")

    # Systolic and Diastolic BP Recommendations
    if systolic_bp is not None and diastolic_bp is not None:
        if systolic_bp > 140 or diastolic_bp > 90:
            recommendations.append("Your blood pressure readings indicate hypertension. Hypertension can increase the risk of heart disease and stroke. It's essential to monitor your blood pressure and follow a healthy lifestyle, including reducing salt intake and managing stress.")
        elif systolic_bp < 90 or diastolic_bp < 60:
            recommendations.append("Your blood pressure is lower than normal. Ensure you are getting adequate nutrients, and if you experience dizziness, consider consulting a healthcare provider.")

    # Exercise Angina Recommendations
    if exerciseAngina is not None:
        if exerciseAngina == 'Yes':
            recommendations.append("You are experiencing chest pain during physical activity, which is a serious concern. It’s crucial to consult a healthcare provider immediately for an accurate diagnosis and proper treatment.")

    # Fasting Blood Sugar Recommendations
    if fastingBS is not None:
        if fastingBS == 'Yes':
            recommendations.append("Your fasting blood sugar level is high, indicating a risk of diabetes. Consider adopting a healthier diet rich in vegetables, whole grains, and lean proteins, and increase physical activity. Regular health checkups are essential to monitor your blood sugar levels.")

    # Max Heart Rate Recommendations
    if maxHR is not None:
        if maxHR > 180:
            recommendations.append("Your maximum heart rate is high, which could indicate cardiovascular stress. It is advisable to reduce the intensity of your workouts and focus on low-impact exercises. Always monitor your heart rate and consult with a healthcare provider for personalized advice.")

    # Resting ECG Recommendations
    if restingECG is not None:
        if restingECG != 'Normal':
            recommendations.append("Your resting ECG result shows abnormalities. This could indicate potential heart problems. Please consult with a healthcare provider for further evaluation and treatment options.")

    # Sleep Duration Recommendations
    if sleep_duration is not None:
        if sleep_duration < 4:
            recommendations.append("Your sleep duration is below 4 hours. Chronic sleep deprivation can affect your mental and physical health. Consider improving your sleep hygiene, avoiding screen time before bed, and establishing a consistent sleep schedule.")
        elif sleep_duration > 9:
            recommendations.append("You are getting more sleep than recommended. Too much sleep can also have negative health effects. Aim for 7-9 hours of sleep per night to maintain optimal health.")
        else:
            recommendations.append("Your sleep duration is within a healthy range. Keep maintaining good sleep habits to ensure continued well-being.")

    # Stress Level Recommendations
    if stress_level is not None:
        if stress_level > 7:
            recommendations.append("Your stress level is high. Chronic stress is linked to various health issues like high blood pressure, heart disease, and mental health disorders. Consider practicing stress management techniques such as yoga, meditation, and mindfulness exercises.")

    # Heart Rate Recommendations
    if heart_rate is not None:
        if heart_rate < 60:
            recommendations.append("Your heart rate is lower than the normal range. A low heart rate could indicate excellent cardiovascular fitness, but if accompanied by dizziness or shortness of breath, consult a healthcare provider.")
        elif heart_rate > 100:
            recommendations.append("Your heart rate is higher than normal. A high resting heart rate could be caused by stress, anxiety, or an underlying health issue. It’s important to consult a healthcare provider for further evaluation and practice relaxation techniques like deep breathing.")
        else:
            recommendations.append("Your heart rate is within the normal range. Keep up with regular exercise and stress management to maintain a healthy heart rate.")

    # BMI Category Recommendations
    if bmi_category is not None:
        if bmi_category == 'Overweight':
            recommendations.append("Your BMI indicates that you are overweight. Being overweight can increase the risk of several health conditions, including heart disease and diabetes. Consider adopting a balanced diet, increasing physical activity, and aiming for a healthy weight through gradual and sustainable changes.")
        elif bmi_category == 'Obese':
            recommendations.append("Your BMI indicates that you are obese. Obesity is associated with serious health risks such as diabetes, heart disease, and joint problems. It’s important to seek advice from a healthcare provider on a personalized weight loss plan that includes healthy eating and exercise.")
        elif bmi_category == 'Underweight':
            recommendations.append("Your BMI indicates that you are underweight. Being underweight can lead to malnutrition and other health issues. Ensure you are eating a well-balanced diet with adequate calories and nutrients, and consult a healthcare provider if necessary.")
        else:
            recommendations.append("Your BMI is within the healthy range. Continue maintaining a balanced diet and regular physical activity to stay healthy.")

    # Daily Steps Recommendations
    if daily_steps is not None:
        if daily_steps < 5000:
            recommendations.append("You are not meeting the recommended 10,000 steps per day. Regular physical activity is key to maintaining a healthy weight, improving cardiovascular health, and reducing stress. Consider increasing your daily activity and finding ways to incorporate more movement into your routine.")
        else:
            recommendations.append("You are meeting or exceeding the recommended daily steps. Keep up the good work in staying active! Regular physical activity can have numerous benefits for your overall health.")

    return recommendations

# Endpoint to retrieve recommendations
@recommendations.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    db_connection = get_db_connection()
    cursor = db_connection.cursor(dictionary=True)

    try:
        # Fetch user data by joining relevant tables
        cursor.execute("""
            SELECT 
                user.name, user.age, user.gender,
                heart.cholesterol, heart.restingBP, heart.fastingBS,
                heart.exerciseAngina, heart.maxHR, heart.restingECG,
                sleep.sleep_duration, sleep.stress_level, sleep.heart_rate,
                sleep.bmi_category, sleep.daily_steps,
                sleep.systolic_bp, sleep.diastolic_bp
            FROM user_data AS user
            LEFT JOIN heart_data AS heart ON user.user_id = heart.user_id
            LEFT JOIN sleep_data AS sleep ON user.user_id = sleep.user_id
            WHERE user.user_id = %s
            ORDER BY sleep.date_recorded DESC LIMIT 1
        """, (user_id,))
        user_data = cursor.fetchone()

        if user_data:
            recommendations_list = generate_recommendations(user_data)
            return jsonify({
                "user": {
                    "name": user_data['name'],
                    "age": user_data['age'],
                    "gender": user_data['gender']
                },
                "recommendations": recommendations_list
            }), 200
        else:
            return jsonify({"error": "User not found or insufficient data"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        db_connection.close()
