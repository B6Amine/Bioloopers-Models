import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the model and label encoder
model = joblib.load('trained_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load the dataset for plotting (replace this with the actual path of your training dataset)
df = pd.read_excel('C:/Users/Amine/Desktop/Training_Data_1000.xlsx')

# Encode the 'calorie_deficit' column
df['calorie_deficit'] = label_encoder.transform(df['calorie_deficit'])

# Define features (X) and target (y)
X = df[['original_weight', 'calorie_deficit', 'training_x_a_week', 'average_distance', 'period_of_training_weeks']]
y = df['weight_change']

# App title
st.title("Weight Change Prediction App")

# User input fields
original_weight = st.number_input("Current weight (in kg)", min_value=30.0, max_value=200.0, step=0.1)
calorie_deficit = st.selectbox("Are you in a calorie deficit?", ['yes', 'no'])
training_x_a_week = st.number_input("How many training sessions per week?", min_value=0, max_value=14, step=1)
average_distance = st.number_input("Average distance per training session (in km)", min_value=0.0, max_value=50.0, step=0.1)
period_of_training_weeks = st.number_input("Duration of the training program (in weeks)", min_value=1, max_value=52, step=1)

# Process the user input to match the LabelEncoder's expected values
calorie_deficit_encoded = label_encoder.transform([calorie_deficit])[0]

# Prepare the data in a DataFrame
user_data = pd.DataFrame({
    "original_weight": [original_weight],
    "calorie_deficit": [calorie_deficit_encoded],
    "training_x_a_week": [training_x_a_week],
    "average_distance": [average_distance],
    "period_of_training_weeks": [period_of_training_weeks]
})

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Predict weight change using the model
    predicted_weight_change = model.predict(user_data)[0]
    
    # Display the predicted result
    st.success(f"Your estimated weight change is: {predicted_weight_change:.2f} kg")

# Button to display the plot based on training data
if st.button("Show Regression Plot"):
    # Generate a range of weights from the dataset for plotting the regression line
    x_range = np.linspace(X['original_weight'].min(), X['original_weight'].max(), 100)
    
    # Create dummy data for the other features based on the mean values from the dataset
    dummy_data = pd.DataFrame({
        'original_weight': x_range,
        'calorie_deficit': [X['calorie_deficit'].mean()] * len(x_range),
        'training_x_a_week': [X['training_x_a_week'].mean()] * len(x_range),
        'average_distance': [X['average_distance'].mean()] * len(x_range),
        'period_of_training_weeks': [X['period_of_training_weeks'].mean()] * len(x_range)
    })
    
    # Predict weight change for each weight in the range
    y_range_pred = model.predict(dummy_data)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X['original_weight'], y, color='blue', label='Training Data')
    plt.plot(x_range, y_range_pred, color='red', label='Regression Line')
    plt.xlabel("Original Weight (kg)")
    plt.ylabel("Weight Change (kg)")
    plt.title("Linear Regression: Original Weight vs Weight Change (Training Data)")
    plt.legend()
    
    # Display the plot in Streamlit
    st.pyplot(plt)
