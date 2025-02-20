from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Serve form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        RM = float(request.form["RM"])
        PTRATIO = float(request.form["PTRATIO"])
        LSTAT = float(request.form["LSTAT"])

        # Create input array with three features
        X_input = np.array([[RM, PTRATIO, LSTAT]])

        # Predict price
        prediction = model.predict(X_input)[0][0]  # Extract scalar value

        return render_template("index.html", prediction_text=f"Predicted Price: ${prediction:.2f}")

    except ValueError:
        return render_template("index.html", prediction_text="Invalid input! Enter valid numbers.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
