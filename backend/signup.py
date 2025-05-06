from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector

# Create a Blueprint for the signup and login functionality
auth = Blueprint('auth', __name__)

# Function to get a database connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',          # Replace with your MySQL username
        password='root123',   # Replace with your MySQL password
        database='health_predictions'  # Ensure it matches your database name
    )

# Define the signup route
@auth.route('/signup', methods=['POST'])
def signup_route():
    try:
        # Get data from the request
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        age = data.get('age')
        gender = data.get('gender')

        # Validate input data
        if not all([name, email, password, age, gender]):
            return jsonify({'error': 'Missing fields in the request'}), 400

        # Check if the email already exists
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM user_data WHERE email = %s', (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            return jsonify({'error': 'Email already exists'}), 409  # Conflict error

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Insert data into the database
        cursor.execute(
            'INSERT INTO user_data (name, email, password, age, gender) VALUES (%s, %s, %s, %s, %s)',
            (name, email, hashed_password, age, gender)
        )
        connection.commit()

        # Close the cursor and connection
        cursor.close()
        connection.close()

        return jsonify({'message': 'Sign Up Successful'}), 200

    except mysql.connector.Error as db_error:
        return jsonify({'error': 'Database error: ' + str(db_error)}), 500
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500

# Define the login route
@auth.route('/login', methods=['POST'])
def login_route():
    try:
        # Get data from the request
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Validate input data
        if not all([email, password]):
            return jsonify({'error': 'Missing email or password'}), 400

        # Check if the email exists in the database
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM user_data WHERE email = %s', (email,))
        user = cursor.fetchone()

        if user is None:
            return jsonify({'error': 'Email not found'}), 404

        # Check if the password matches the hashed password
        if not check_password_hash(user[3], password):  # user[3] is the password column
            return jsonify({'error': 'Invalid password'}), 401

        # Set user_id in session after successful login
        session['user_id'] = user[0]  # user[0] is the user_id from the database

        # Close the cursor and connection
        cursor.close()
        connection.close()

        return jsonify({'message': 'Login Successful', 'user_id': user[0]}), 200

    except mysql.connector.Error as db_error:
        return jsonify({'error': 'Database error: ' + str(db_error)}), 500
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500




