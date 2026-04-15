import streamlit as st
import pickle
import numpy as np

# ============================================
# Load Trained Model
# ============================================
model = pickle.load(open("model.pkl", "rb"))

# ============================================
# App Title
# ============================================
st.title("🚗 Road Accident Severity Prediction")
st.write("Enter accident details to predict severity")

# ============================================
# User Inputs (EDIT based on your dataset)
# ============================================

weather = st.selectbox(
    "Weather Condition",
    ["Clear", "Rain", "Fog"]
)

road_type = st.selectbox(
    "Road Type",
    ["Single Carriageway", "Dual Carriageway", "Roundabout"]
)

light = st.selectbox(
    "Light Condition",
    ["Daylight", "Dark", "Street Lighting"]
)

# ============================================
# Convert Inputs to Numerical (IMPORTANT)
# ============================================

weather_map = {"Clear": 0, "Rain": 1, "Fog": 2}
road_map = {"Single Carriageway": 0, "Dual Carriageway": 1, "Roundabout": 2}
light_map = {"Daylight": 0, "Dark": 1, "Street Lighting": 2}

input_data = np.array([[
    weather_map[weather],
    road_map[road_type],
    light_map[light]
]])

# ============================================
# Prediction
# ============================================

if st.button("Predict Severity"):
    prediction = model.predict(input_data)

    # Map output to labels (EDIT based on your dataset)
    severity_map = {
        0: "Slight",
        1: "Serious",
        2: "Fatal"
    }

    result = severity_map.get(prediction[0], "Unknown")

    st.success(f"Predicted Accident Severity: {result}")