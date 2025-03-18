import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Streamlit Title and Instructions
st.title("🔧 Prediction of Motor Health")
st.write("Upload your sensor data (CSV format) and check the motor's health!")

# File Uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Load the Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("motor_model.keras")
    return model

model = load_model()

# Preprocessing Function
def preprocess_data(df):
    # Drop unnecessary columns
    cols_to_drop = ['UDI', 'Product ID', 'Type']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    
    if existing_cols:
        df = df.drop(existing_cols, axis=1)

    # Ensure the CSV columns match the model input
    expected_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                     'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    if not all(col in df.columns for col in expected_cols):
        st.error("❌ CSV columns do not match expected format.")
        return None

    # Select and reorder columns to match the model input
    df = df[expected_cols]

    # Normalize the data
    mean = np.array([298, 310, 1500, 40, 200, 0, 0, 0, 0, 0, 0])
    std = np.array([10, 15, 500, 20, 100, 1, 1, 1, 1, 1, 1])

    # Apply normalization
    data_normalized = (df.values - mean) / std

    return data_normalized, df

# Prediction Function
def make_predictions(data):
    predictions = model.predict(data)
    predictions_sigmoid = 1 / (1 + np.exp(-predictions))

    threshold = 0.5
    results = ["Healthy" if p < threshold else "Faulty" for p in predictions_sigmoid.flatten()]
    return results

# Handle Uploaded CSV File
if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Preprocess and Predict
    result = preprocess_data(df)

    if result is not None:
        data_normalized, processed_df = result
        predictions = make_predictions(data_normalized)

        # Display Results
        st.write("### ✅ Predictions:")
        for i, pred in enumerate(predictions, start=1):
            st.write(f"Sample {i}: **{pred}**")

        # Display Summary
        healthy_count = predictions.count("Healthy")
        faulty_count = predictions.count("Faulty")

        st.write(f"✅ **Healthy samples:** {healthy_count}")
        st.write(f"❌ **Faulty samples:** {faulty_count}")

