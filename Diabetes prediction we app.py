# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:58:45 2025

@author: sachin
"""
import numpy as np
import pickle 
import streamlit as st

# Load the saved model
model_path = 'trained_model.sav'
loaded_model = pickle.load(open(model_path, 'rb'))  # rb -> read binary

# Function for Prediction
def Diabetes_prediction(input_data):
    # Convert input data to a NumPy array (ensure all values are numeric)
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array for a single instance prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Interpret result
    if prediction[0] == 0:
        return 'The person is NOT diabetic'
    else:
        return 'The person is DIABETIC'

# Main function for Streamlit App
def main():
    # Title
    st.title('Diabetes Prediction Web App')

    # Get user input
    Pregnancies = st.number_input('Number of pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0.0)
    BloodPressure = st.number_input('Blood Pressure Value', min_value=0.0)
    SkinThickness = st.number_input('Skin Thickness Value', min_value=0.0)
    Insulin = st.number_input('Insulin Level', min_value=0.0)
    BMI = st.number_input('BMI Value', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value', min_value=0.0)
    Age = st.number_input('Age of the person', min_value=0, step=1)

    # Prediction button
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = Diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

# Run the app
if __name__ == '__main__':
    main()
