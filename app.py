from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app, origins=[
    "https://health-plus2.vercel.app"
])

# Load trained model
model = joblib.load("model/diabetes_model.pkl")

@app.route("/")
def home():
    return "HealthPlus AI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            data["Pregnancies"],
            data["Glucose"],
            data["BloodPressure"],
            data["SkinThickness"],
            data["Insulin"],
            data["BMI"],
            data["DiabetesPedigreeFunction"],
            data["Age"]
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if probability < 0.30:
            risk = "Low Risk"
        elif probability < 0.70:
            risk = "Moderate Risk"
        else:
            risk = "High Risk"

        return jsonify({
            "prediction": risk,
            "probability": round(float(probability) * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)