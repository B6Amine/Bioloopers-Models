import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the model, encoder, and scaler
model = joblib.load('ridge_regression_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Athlete Performance Prediction App")

# User input fields
height = st.number_input("Height (in cm)", min_value=140.0, max_value=210.0, step=0.1)
age = st.number_input("Age", min_value=16, max_value=70, step=1)
gender = st.selectbox("Gender", ['male', 'female'])
background = st.selectbox("Athletic Background", ['Youth/High school sports', 'College sports', 'No background', 'Professional sports'])
schedule = st.selectbox("Training Schedule", ['Single workout per day', 'Multiple workouts 2x a week', 'Multiple workouts 3+ times a week'])
train = st.selectbox("Training Location", ['CrossFit Affiliate', 'With Coach', 'Alone'])
howlong = st.number_input("How long have you been training (in years)?", min_value=0.5, max_value=20.0, step=0.1)

# Prepare the user input for prediction
user_data = pd.DataFrame({
    "height": [height],
    "age": [age],
    "gender": [1 if gender == 'male' else 0],
    "howlong": [howlong],
    "background": [background],
    "schedule": [schedule],
    "train": [train]
})

# Perform one-hot encoding for the categorical variables
user_encoded = encoder.transform(user_data[['background', 'schedule', 'train']])

# Standardize the numerical variables
user_scaled = scaler.transform(user_data[['height', 'age', 'gender', 'howlong']])

# Combine the scaled and encoded user inputs
user_final = np.hstack([user_scaled, user_encoded])

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Predict athletic performance using the model
    predicted_performance = model.predict(user_final)
    
    # Output the prediction for each exercise
    exercises = ['Fran', 'Helen', 'Grace', 'Filthy50', 'Fight Gone Bad', '400m Run', '5k Run', 'Clean & Jerk', 'Snatch', 'Deadlift', 'Back Squat', 'Pullups']
    st.subheader("Predicted Performance:")
    for i, exercise in enumerate(exercises):
        st.write(f"{exercise}: {predicted_performance[0, i]:.2f}")

# Button to display the plot based on the predicted performance
if st.button("Show Performance Plot"):
    exercises = ['Fran', 'Helen', 'Grace', 'Filthy50', 'Fight Gone Bad', '400m Run', '5k Run', 'Clean & Jerk', 'Snatch', 'Deadlift', 'Back Squat', 'Pullups']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(exercises, predicted_performance[0], color='blue')
    plt.xlabel('Exercise')
    plt.ylabel('Predicted Performance')
    plt.title('Predicted Athletic Performance Across Exercises')
    plt.xticks(rotation=45)
    
    # Display the plot in Streamlit
    st.pyplot(plt)
