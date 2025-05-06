from flask import Flask,session, jsonify
from flask_cors import CORS  # Import CORS
from datetime import timedelta 
from flask_sqlalchemy import SQLAlchemy
from signup import auth  # Import signup and login routes
from heart import heart # Import heart prediction routes
from diabetes import diabetes
from sleep import sleep
from dashboard import dashboard
from recommendations import recommendations
from pcos import pcos
app = Flask(__name__)
app.secret_key = 'root456'  
app.permanent_session_lifetime = timedelta(days=700) 
app.config['SESSION_TYPE'] = 'sqlalchemy'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:root123@localhost/health_predictions'  # Replace with your database credentials
app.config['SESSION_SQLALCHEMY'] = SQLAlchemy(app)  
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True  
app.config['SESSION_KEY_PREFIX'] = 'health_app:'

CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return 'Welcome to the Health App'
# Register blueprints for signup, login, and heart prediction
app.register_blueprint(auth, url_prefix='/auth')  # Register auth routes
@app.route('/check_session')
def check_session():
    if 'user_id' in session:
        return jsonify({'message': f'User is logged in. User ID: {session["user_id"]}'}), 200
    else:
        return jsonify({'error': 'No user logged in'}), 401 
    
app.register_blueprint(heart, url_prefix='/heart')  
app.register_blueprint(diabetes, url_prefix='/diabetes') 
app.register_blueprint(sleep, url_prefix='/sleep') 
app.register_blueprint(pcos, url_prefix='/pcos') 
app.register_blueprint(dashboard, url_prefix='/dashboard') 
app.register_blueprint(recommendations , url_prefix='/recommendations')
if __name__ == '__main__':
    app.run(debug=True)



   


