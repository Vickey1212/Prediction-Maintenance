import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import time
import random  # Simulate IoT sensor data

# Constants for model storage
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "label_encoders.pkl"
TARGET_ENCODER_FILE = "target_encoder.pkl"
FEATURES_FILE = "feature_columns.pkl"

# Load model and preprocessing objects
@st.cache_resource
def load_saved_objects():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    label_encoders = joblib.load(ENCODERS_FILE)
    target_encoder = joblib.load(TARGET_ENCODER_FILE)
    feature_columns = joblib.load(FEATURES_FILE)
    return model, scaler, label_encoders, target_encoder, feature_columns

model, scaler, label_encoders, target_encoder, feature_columns = load_saved_objects()

# Title
st.title("🌐 Real-Time Predictive Maintenance System")
st.write("📡 Monitoring live IoT sensor data for failure detection.")

# Function to simulate sensor data
def simulate_sensor_data():
    sensor_data = {
        "Air Temperature [K]": round(random.uniform(290, 340), 2),
        "Process Temperature [K]": round(random.uniform(290, 370), 2),
        "Rotational Speed [rpm]": round(random.uniform(500, 5000), 2),
        "Torque [Nm]": round(random.uniform(2, 250), 2),
        "Tool Wear [min]": round(random.uniform(0, 600), 2),
    }
    if random.random() < 0.5:
        sensor_data["Torque [Nm]"] = round(random.uniform(200, 250), 2)
        sensor_data["Tool Wear [min]"] = round(random.uniform(550, 600), 2)
    return sensor_data

# Highlighting function
def highlight_value(feature, value):
    red_conditions = {
        "Air Temperature [K]": value > 330,
        "Process Temperature [K]": value > 350,
        "Torque [Nm]": value > 200,
        "Tool Wear [min]": value > 550,
        "Rotational Speed [rpm]": value > 4000,
    }
    color = "red" if red_conditions.get(feature, False) else "green"
    return f"<span style='color:{color}; font-weight:bold'>{value}</span>"

# Placeholder containers
sensor_placeholder = st.empty()
status_placeholder = st.empty()
chart_placeholder = st.empty()

# --- Simulation Execution ---
sensor_data = simulate_sensor_data()

# Convert to DataFrame
new_data = pd.DataFrame([sensor_data], columns=feature_columns)
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
predicted_status = target_encoder.inverse_transform(prediction)[0]

# --- Sensor Table (Styled CSS Grid) ---
with sensor_placeholder.container():
    st.subheader("📊 Live Sensor Readings")

    custom_css = """
    <style>
        .sensor-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            max-width: 600px;
        }
        .sensor-box {
            background: #121212;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #444;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """

    sensor_html = "<div class='sensor-container'>"
    for feature, value in sensor_data.items():
        colored_value = highlight_value(feature, value)
        sensor_html += f"<div class='sensor-box'>{feature}<br>{colored_value}</div>"
    sensor_html += "</div>"

    st.markdown(custom_css + sensor_html, unsafe_allow_html=True)

# --- Prediction Result ---
with status_placeholder.container():
    st.subheader("📢 Machine Status Prediction")
    if predicted_status == "No Failure":
        st.success("✅ Machine is operating normally.")
    else:
        st.error(f"⚠ *Failure Detected: {predicted_status}*")

        # SHAP Explainability
        explainer = shap.Explainer(model, new_data_scaled)
        shap_values = explainer(new_data_scaled)
        importance_values = np.abs(shap_values.values).mean(axis=0)
        most_important_features = np.array(feature_columns)[np.argsort(-importance_values)][:3]

        st.write("🔍 *Possible Causes of Failure:*")
        for feature in most_important_features:
            st.write(f"➡ *{feature}*: Unusual value detected")

        # SHAP Chart
        with chart_placeholder.container():
            st.write("📊 *Feature Contribution to Failure*")
            fig, ax = plt.subplots(figsize=(8, 5))
            shap_df = pd.DataFrame({
                "Feature": feature_columns,
                "SHAP Value": importance_values
            }).sort_values("SHAP Value", ascending=False)
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color="red")
            ax.set_xlabel("SHAP Value (Impact on Model Prediction)")
            ax.set_title("Feature Contribution to Failure")
            st.pyplot(fig)

# Refresh every 5 seconds
time.sleep(5)
st.experimental_rerun()
