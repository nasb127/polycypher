from flask import Flask, request, jsonify
import dill
import joblib
import numpy as np

app = Flask(__name__)

# Load your model and scaler
with open('pcos_model.pkl', 'rb') as f:
    model = dill.load(f)

scaler = joblib.load('scaler.pkl')

# 🛠️ Fix: Define a homepage route for health checks and users
@app.route("/")
def home():
    return "<h1>✅ PCOS Predictor API is Running</h1><p>Use the /predict endpoint to POST data.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['features']])
    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
