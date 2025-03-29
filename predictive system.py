# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script for predicting diabetes using a trained ML model.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open(r'C:/Users/shrey/Desktop/code/ML/dataset/project/Diabetes Prediction/trained_model.pkl', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

#charging the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicitng for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print('The Person is not diabetic')
else:
    print('The person is Diabetic')
