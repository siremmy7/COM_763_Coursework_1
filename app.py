import streamlit as st
import pickle
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Real Estate Predictor", page_icon="🏠")

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ============================================
# TITLE
# ============================================
st.title("🏠 Real Estate Price Prediction System")
st.markdown("Predict house prices using a trained Random Forest model.")

# ============================================
# SIDEBAR INPUTS
# ============================================
st.sidebar.header("Enter Property Details")

x1 = st.sidebar.number_input("Transaction Date", value=2013.0)
x2 = st.sidebar.slider("House Age (years)", 0.0, 50.0, 10.0)
x3 = st.sidebar.number_input("Distance to MRT (meters)", value=300.0)
x4 = st.sidebar.slider("Number of Convenience Stores", 0, 15, 5)
x5 = st.sidebar.number_input("Latitude", value=24.97)
x6 = st.sidebar.number_input("Longitude", value=121.54)

# ============================================
# PREPARE INPUT DATA
# ============================================
input_data = np.array([[x1, x2, x3, x4, x5, x6]])

# ============================================
# DISPLAY INPUT SUMMARY
# ============================================
st.subheader("📊 Input Summary")
st.write({
    "Transaction Date": x1,
    "House Age": x2,
    "Distance to MRT": x3,
    "Convenience Stores": x4,
    "Latitude": x5,
    "Longitude": x6
})

# ============================================
# PREDICTION BUTTON
# ============================================
if st.button("🔍 Predict Price"):
    try:
        prediction = model.predict(input_data)

        st.success(f"💰 Estimated House Price: {prediction[0]:.2f}")

    except Exception as e:
        st.error("❌ Error making prediction. Check input format.")
        st.write(e)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("Developed for COM763 - Advanced Machine Learning Coursework")