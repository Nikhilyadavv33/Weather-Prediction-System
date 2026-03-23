from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
from tensorflow.keras.models import load_model

lstm_model = load_model("lstm_model.h5", compile=False)

# Load models
model = load_model("model.h5")
lr_model = joblib.load("lr_model.pkl")  # optional (not displayed)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = []

    for x in request.form.values():
        if x == "":
            return render_template('index.html', prediction_text="⚠️ Please fill all fields")
        data.append(float(x))

    dnn_pred = model.predict(np.array([data]))[0][0]

    prob = dnn_pred * 100

    if dnn_pred > 0.5:
        result = f"🌧️ Rain Expected ({prob:.2f}%)"
    else:
        result = f"☀️ No Rain ({100 - prob:.2f}%)"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)