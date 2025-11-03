import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# Create app - templates are in the 'templates' folder in the same directory
app = Flask(__name__)

# Diagnostic logging
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
print("Templates directory exists?:", os.path.exists('templates'))
if os.path.exists('templates'):
    print("Files in templates:", os.listdir('templates'))

# Global model/scaler (lazy-loaded)
model = None
scaler = None


def get_model():
    
    global model, scaler
    if model is None or scaler is None:
        print("Loading model and scaler...")
        try:
            # Load XGBoost model
            model = xgb.Booster()
            model.load_model('xgb_breastcare.json')
            print(" XGBoost model loaded successfully!")

            # Load scaler
            scaler = joblib.load('scaler.pkl')
            print(" Scaler loaded successfully!")

        except Exception as e:
            print(f" Error loading models: {e}")
            raise e
    return model, scaler


@app.route("/ping")
def ping():
    return {"status": "ok", "message": "Flask app is alive!"}


@app.route("/")
def home():
    return render_template("home.html", query="")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["POST"])
def cancerPrediction():
    try:
        # Get inputs from form
        inputQuery1 = float(request.form['query1'])
        inputQuery2 = float(request.form['query2'])
        inputQuery3 = float(request.form['query3'])
        inputQuery4 = float(request.form['query4'])
        inputQuery5 = float(request.form['query5'])

        # Prepare input
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
        new_df = pd.DataFrame(data, columns=[
            'perimeter_worst', 'concave points_worst', 'concave points_mean',
            'area_mean', 'area_worst'
        ])

        # Load model/scaler
        model, scaler = get_model()

        # Scale & predict
        new_df_scaled = scaler.transform(new_df)
        dtest = xgb.DMatrix(new_df_scaled)
        proba = model.predict(dtest)  # probabilities

        # XGBoost Booster returns probability of class 1 if binary
        pred_class = int(proba[0] >= 0.5)

        if pred_class == 1:
            output1 = "The patient is diagnosed with Breast Cancer"
            confidence = float(proba[0] * 100)
        else:
            output1 = "The patient is not diagnosed with Breast Cancer"
            confidence = float((1 - proba[0]) * 100)

        output2 = f"Confidence: {confidence:.2f}%"
        return render_template("home.html", output1=output1, output2=output2,
                               query1=request.form['query1'], query2=request.form['query2'],
                               query3=request.form['query3'], query4=request.form['query4'],
                               query5=request.form['query5'])

    except Exception as e:
        error_message = f"Error: {e}"
        print(f"Prediction error: {e}")
        return render_template("home.html", output1=error_message, output2="")


# For Railway deployment
if __name__ == "__main__":
    # Pre-load models to check if they work
    try:
        get_model()
        print("Models loaded successfully during startup!")
    except Exception as e:
        print(f"Model loading failed: {e}")

    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
