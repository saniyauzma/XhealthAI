XhealthAI is a health-focused AI web application that provides predictive analytics and recommendations for various health conditions, including diabetes, heart disease, PCOS, and sleep disorders. The platform leverages machine learning models to assist users in understanding their health risks and offers personalized recommendations.

## Features
- Predictive analytics for:
  - Diabetes
  - Heart Disease
  - PCOS (Polycystic Ovary Syndrome)
  - Sleep Disorders
- Personalized health recommendations
- User signup and authentication
- Interactive dashboard
- Chatbot for user queries
- Informative resources and contact page

## Project Structure
```
XhealthAI-main/
  backend/           # Backend Python code and ML models
    app.py           # Main backend application
    dashboard.py     # Dashboard logic
    diabetes.py      # Diabetes prediction logic
    heart.py         # Heart disease prediction logic
    pcos.py          # PCOS prediction logic
    sleep.py         # Sleep disorder prediction logic
    recommendations.py # Health recommendations
    signup.py        # User signup/authentication
    model/           # ML models for heart
    modeld/          # ML models for diabetes
    modelp/          # ML models for PCOS
    models/          # Additional models (e.g., sleep)
  css/               # Stylesheets for frontend
  html/              # HTML templates/pages
  images/            # Image assets
  README.md          # Project documentation
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd XhealthAI-main/XhealthAI-main
   ```
2. **Set up a Python virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MySQL:**
   - Install MySQL and ensure it is running.
   - Create a database named `health_predictions` and required tables (e.g., `user_data`).
   - Update credentials in `backend/app.py` and related files if needed.

5. **Obtain Model Files:**
   - Use the provided Google Colab notebooks (e.g., `heartfinal.ipynb`) to train and export model files as `.pkl`.
   - In Colab, after training, download each model file using:
     ```python
     from google.colab import files
     files.download('logistic_regression_model.pkl')
     # Repeat for each .pkl file
     ```
   - Place the downloaded `.pkl` files in the correct folders:
     - `backend/model/` for heart models
     - `backend/modeld/` for diabetes models
     - `backend/modelp/` for PCOS models
     - `backend/models/` for sleep models

6. **Run the backend server:**
   ```bash
   python backend/app.py
   ```

7. **Open the frontend:**
   - Open the HTML files in the `html/` directory in your browser, or set up a simple HTTP server to serve them.

## Usage
- Access the main dashboard via `dashboard.html` after starting the backend.
- Use the prediction forms for diabetes, heart disease, PCOS, and sleep analysis.
- View personalized recommendations and interact with the chatbot for more information.

## Troubleshooting
- **Model file not found:**
  - Ensure all `.pkl` files are in the correct folders as described above.
  - The backend must be run from the `XhealthAI-main/XhealthAI-main` directory.
- **Model downloads as .py file:**
  - In Colab, use `files.download('your_model.pkl')` after saving with `joblib.dump(...)`.
  - The downloaded file should have a `.pkl` extension.
- **MySQL errors:**
  - Ensure MySQL is running and the database/tables exist.
  - Check credentials in the backend code.

## Pushing to GitHub
1. **Initialize git (if not already):**
   ```bash
   git init
   git remote add origin <your-repo-url>
   ```
2. **Add and commit changes:**
   ```bash
   git add .
   git commit -m "Update README and project setup instructions"
   ```
3. **Push to GitHub:**
   ```bash
   git push -u origin main
   ```
   *(Replace `main` with your branch name if different)*

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

