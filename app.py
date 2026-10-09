from flask import Flask, request, render_template
import numpy as np
import joblib


# -----------------------------
# Load trained model
# -----------------------------
obj = joblib.load("California.joblib")

model = obj["Model"]
columns = obj["Columns"]

print("Model columns:", columns)


# -----------------------------
# Create Flask application
# -----------------------------
app = Flask(__name__)


# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def main():
    return render_template("index.html", columns=columns)


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    Input = []

    # Get values from HTML form
    for column in columns:

        value = request.form.get(column)

        # Convert input to float
        value = float(value)

        Input.append(value)

    # Make prediction
    out = model.predict([Input])

    # Convert numpy value to normal Python number
    prediction = float(out[0])

    return render_template(
        "index.html",
        columns=columns,
        prediction=prediction
    )


# -----------------------------
# Run Flask application
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)