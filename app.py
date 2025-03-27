import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the saved model

# Load the trained model
model = joblib.load("loan_default_model.pkl")  # Make sure you save your trained model

# Streamlit UI
st.title("Economic Well-Being Prediction")

# User Inputs
st.write("Enter the feature values to predict the target:")
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)

# Convert inputs into DataFrame
input_data = pd.DataFrame([[feature_1, feature_2, feature_3]], columns=["Feature 1", "Feature 2", "Feature 3"])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Target: {prediction[0]:.4f}")


