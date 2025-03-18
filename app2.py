import streamlit as st

st.title("Prediction of motor health")
st.write("Welcome! Upload your sensor data and check the motor's health.")

# Add file uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
