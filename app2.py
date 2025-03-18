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
    """
    Preprocess the CSV data:
    - Drop 'UDI' and 'Product ID'.
    - One-hot encode 'Type' (M -> 0, F -> 1).
    - Ensure proper column order.
    - Normalize the data using model's training mean and std.
    """
    # 🚫 Drop irrelevant columns
    df = df.drop(['UDI', 'Product ID'], axis=1)

    # 🔥 One-hot encode 'Type' column
    df['Type'] = df['Type'].map({'M': 0, 'F': 1})

    # ✅ Ensure correct column order (12 columns)
    expected_cols = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                     'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    # Check for missing columns
    if not all(col in df.columns for col in expected_cols):
        st.error("❌ CSV columns do not match expected format.")
        return None, None

    # Reorder columns
    df = df[expected_cols]

    # 🔥 Normalize the data
    mean = np.array([0.5, 298, 310, 1500, 40, 200, 0, 0, 0, 0, 0, 0])
    std = np.array([0.5, 10, 15, 500, 20, 100, 1, 1, 1, 1, 1, 1])

    # Apply normalization
    data_normalized = (df.values - mean) / std

    return data_normalized, df

# Prediction Function
def make_predictions(data):
    """Generate predictions from the model and classify results."""
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
        st.write(f"Model expects shape: (n, 12)")
        st.write(f"Preprocessed data shape: {data_normalized.shape}")
        
        predictions = make_predictions(data_normalized)

        # Display Predictions
        st.write("### ✅ Predictions:")
        result_df = df.copy()
        result_df['Prediction'] = predictions
        st.dataframe(result_df)

        # Downloadable CSV with Predictions
        csv_output = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv_output, file_name="motor_health_predictions.csv", mime="text/csv")

