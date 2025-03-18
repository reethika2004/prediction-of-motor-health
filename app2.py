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
    # Drop the UDI column if it exists
    if 'UDI' in df.columns:
        df = df.drop(columns=['UDI'])

    # Ensure the CSV columns match the model input
    expected_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                     'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF',
                     'Type_L', 'Type_M', 'Type_H']  # Modify column names if needed

    if not all(col in df.columns for col in expected_cols):
        st.error("❌ CSV columns do not match expected format.")
        return None

    # Select and reorder columns to match the model input
    df = df[expected_cols]

    # Normalize the data
    mean = np.array([298, 310, 1500, 40, 200, 0, 0, 0, 0, 0, 0, 0])
    std = np.array([10, 15, 500, 20, 100, 1, 1, 1, 1, 1, 1, 1])

    # Apply normalization
    data_normalized = (df.values - mean) / std
    return data_normalized, df

# Prediction Function
def make_predictions(data):
    predictions = model.predict(data)
    predictions_sigmoid = 1 / (1 + np.exp(-predictions))

    # Classify based on sigmoid output
    results = ["Healthy" if p < 0.5 else "Faulty" for p in predictions_sigmoid.flatten()]
    return results

# Handle Uploaded CSV File
if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Preprocess and Predict
    data_normalized, processed_df = preprocess_data(df)

    if data_normalized is not None:
        predictions = make_predictions(data_normalized)

        # Display Predictions
        st.write("### ✅ Predictions:")
        result_df = df.copy()  # Include the UDI column if it exists
        result_df['Prediction'] = predictions
        st.dataframe(result_df)

        # Downloadable CSV with Predictions
        csv_output = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv_output, file_name="motor_health_predictions.csv", mime="text/csv")
