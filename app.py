import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ltn
import tensorflow as tf

# Load the dataset
feeds_df = pd.read_csv("feeds2.csv")

# Data preprocessing
feeds_df.dropna(inplace=True)
feeds_df['rotational_speed'] = feeds_df['rotational_speed'].apply(lambda x: min(x, 250))  # Scaling rotational speed
feeds_df['air_temperature'] = (feeds_df['air_temperature'] - feeds_df['air_temperature'].min()) / \
                              (feeds_df['air_temperature'].max() - feeds_df['air_temperature'].min()) * 50  # Scaling air temperature up to 50
X = feeds_df[['temperature', 'humidity', 'vibration', 'rotational_speed', 'air_temperature']]

# Train-test split
y = (feeds_df['vibration'] < 1) & (feeds_df['temperature'] < 65)
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# SHAP Analysis
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test)

# Streamlit UI
st.title("Motor Health Prediction using LTN & XAI")

# User input for prediction
st.sidebar.header("Enter Motor Parameters")
temp = st.sidebar.slider("Temperature", float(X['temperature'].min()), float(X['temperature'].max()), 50.0)
humidity = st.sidebar.slider("Humidity", float(X['humidity'].min()), float(X['humidity'].max()), 50.0)
vibration = st.sidebar.slider("Vibration", float(X['vibration'].min()), float(X['vibration'].max()), 1.0)
speed = st.sidebar.slider("Rotational Speed", float(X['rotational_speed'].min()), float(X['rotational_speed'].max()), 200.0)
air_temp = st.sidebar.slider("Air Temperature", float(X['air_temperature'].min()), float(X['air_temperature'].max()), 30.0)

# Prediction function
def predict_motor_status(temp, humidity, vib, speed, air_temp):
    test_input = np.array([[temp, humidity, vib, speed, air_temp]])
    prediction = clf.predict(test_input)
    return "Motor is GOOD ✅" if prediction == 1 else "Motor is BAD ❌"

if st.sidebar.button("Predict"):
    result = predict_motor_status(temp, humidity, vibration, speed, air_temp)
    st.sidebar.write(result)

# Display SHAP Summary Plot
st.subheader("SHAP Feature Importance")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(fig)
