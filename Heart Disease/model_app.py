import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load model and preprocessing objects
model = joblib.load("decison.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title and description
st.title("❤️ Heart Disease Prediction Model")
st.markdown("Predict the likelihood of heart disease based on medical parameters")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Information")
    
    age = st.slider("Age (years)", 18, 100, 40)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex_encoded = 1 if sex == "Male" else 0
    
    st.subheader("🩺 Medical Parameters")
    
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar (>120 mg/dl)", [0, 1], 
                       format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)

with col2:
    st.subheader("📊 Model Information")
    st.info("""
    This model uses Decision Tree algorithm to predict:
    - **0**: No heart disease
    - **1**: Heart disease present
    
    Features used:
    - Age
    - Sex
    - Cholesterol
    - Fasting Blood Sugar
    - Oldpeak (ST depression)
    """)

# Prediction button
if st.button("🔍 Make Prediction", use_container_width=True):
    try:
        # Create input dataframe with correct column order
        input_df = pd.DataFrame({
            "age": [age],
            "sex": [sex_encoded],
            "chol": [chol],
            "fbs": [fbs],
            "oldpeak": [oldpeak]
        })
        
        # Scale only the columns that were scaled during training
        scaled_cols = ["age", "chol", "oldpeak"]
        input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])
        
        # Ensure columns match the training order
        input_df = input_df[columns]
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Display results
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 0:
                st.success("✅ **No Heart Disease Detected**")
                st.metric("Risk Level", "Low", 
                         f"Confidence: {prediction_proba[0][0]*100:.2f}%")
            else:
                st.error("⚠️ **Heart Disease Risk Detected**")
                st.metric("Risk Level", "High",
                         f"Confidence: {prediction_proba[0][1]*100:.2f}%")
        
        with col2:
            st.subheader("Prediction Confidence")
            confidence_data = {
                "No Disease": prediction_proba[0][0] * 100,
                "Disease": prediction_proba[0][1] * 100
            }
            st.bar_chart(confidence_data)
        
        # Input summary
        st.subheader("📝 Input Summary")
        summary_df = pd.DataFrame({
            "Parameter": ["Age", "Sex", "Cholesterol", "Fasting Blood Sugar", "Oldpeak"],
            "Value": [f"{age} years", sex, f"{chol} mg/dl", 
                     "Yes" if fbs == 1 else "No", f"{oldpeak}"]
        })
        st.table(summary_df)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please ensure all inputs are valid.")

# Footer
st.divider()
st.markdown("⚕️ **Disclaimer**: This model is for educational purposes only and should not be used for actual medical diagnosis.")
