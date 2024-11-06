# ANN-Classification-churn
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

geo_encoded = onehot_geography.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geography.get_feature_names_out(['Geography']))
geo_encoded_df

# Convert the input_data dictionary to a DataFrame first
input_data = pd.DataFrame([input_data])

# Now you can use reset_index and other DataFrame methods
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data = input_data.drop('Geography', axis=1)

input_data['Gender'] = encode_gender.transform(input_data['Gender'])

scaled_input_data = StandardScaler()
input_scaled_df = scaled_input_data.fit_transform(input_data)


##Integrating ANN model with Streamlit Web App

%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('encode_gender.pkl', 'rb') as f:
    encode_gender = pickle.load(f)

with open('onehot_geography.pkl', 'rb') as f:
    onehot_geography = pickle.load(f)

with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)  # Change 'Scalar' to 'scaler' for clarity

# Streamlit App
st.title('Customer Churn Prediction')

# User Inputs
geography = st.selectbox('Geography', onehot_geography.categories_[0])
gender = st.selectbox('Gender', encode_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': encode_gender.transform([gender]),
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# One-hot encode 'Geography' using selected value
geo_encoded = onehot_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geography.get_feature_names_out(['Geography']))

# Combine encoded 'Geography' with other inputs
input_df = pd.concat([pd.DataFrame(input_data), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn probability: {prediction_proba: .2f}')

# Display result
if prediction_proba > 0.5:
    st.write("Customer will exit")
else:
    st.write("Customer will not exit")
