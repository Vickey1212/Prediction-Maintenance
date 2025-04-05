from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import numpy as np

# Load the saved model and preprocessors
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    failure_status_encoder = joblib.load("failure_status_encoder.pkl")
    label_encoders = joblib.load("label_encoders.pkl")  # For categorical features
    feature_columns = joblib.load("feature_columns.pkl")  # Expected feature names
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit("Missing model or preprocessing files. Ensure all .pkl files are present.")

app = Flask(__name__)

# HTML Page with Bootstrap
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Failure Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .container { max-width: 600px; margin-top: 50px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .btn-primary { width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center">Machine Failure Prediction</h2>
        <form method="POST" action="/">
            <div class="mb-3">
                <label class="form-label">Air Temperature (°C)</label>
                <input type="number" step="any" name="Air Temperature" class="form-control" required>
            </div>
            <div class="mb-3">
                <label class="form-label">Process Temperature (°C)</label>
                <input type="number" step="any" name="Process Temperature" class="form-control" required>
            </div>
            <div class="mb-3">
                <label class="form-label">Rotational Speed (RPM)</label>
                <input type="number" step="any" name="Rotational Speed" class="form-control" required>
            </div>
            <div class="mb-3">
                <label class="form-label">Torque (Nm)</label>
                <input type="number" step="any" name="Torque" class="form-control" required>
            </div>
            <div class="mb-3">
                <label class="form-label">Tool Wear (min)</label>
                <input type="number" step="any" name="Tool Wear" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-primary">Predict</button>
        </form>

        {% if prediction %}
        <div class="mt-4 p-3 bg-light border rounded">
            <h4>Prediction:</h4>
            <p><strong>Failure Status:</strong> {{ prediction }}</p>
            <p><strong>Failure Type:</strong> {{ failure_type }}</p>
        </div>
        {% endif %}

        {% if error %}
        <div class="mt-4 p-3 bg-danger text-white border rounded">
            <h4>Error:</h4>
            <p>{{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get input data from form
            input_data = {}
            for key in request.form:
                value = request.form[key]
                if key in label_encoders:  # If categorical, encode it
                    input_data[key] = label_encoders[key].transform([value])[0]
                else:
                    input_data[key] = float(value)  # Convert numerical values

            # Convert to DataFrame and ensure correct column order
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            # Scale numerical features
            input_scaled = scaler.transform(input_df)

            # Make a prediction
            prediction = model.predict(input_scaled)
            predicted_status = target_encoder.inverse_transform(prediction)[0]
            failure_type = failure_status_encoder.inverse_transform(prediction)[0]

            return render_template_string(HTML_PAGE, prediction=predicted_status, failure_type=failure_type)

        except Exception as e:
            return render_template_string(HTML_PAGE, error=str(e))

    return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    app.run(debug=True)
