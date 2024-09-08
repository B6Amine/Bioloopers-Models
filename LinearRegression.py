import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

#  Train the model 
def train_model():
    file_path = "C:/Users/Amine/Desktop/Training_Data_1000.xlsx"  # Update this with the actual path to your Excel file
    df = pd.read_excel(file_path)

    # Encode the 'calorie_deficit' column
    le = LabelEncoder()
    df['calorie_deficit'] = le.fit_transform(df['calorie_deficit'])

    # Define features (X) and target (y)
    X = df[['original_weight', 'calorie_deficit', 'training_x_a_week', 'average_distance', 'period_of_training_weeks']]
    y = df['weight_change']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, X_train, y_train)

    # Save the trained model and label encoder
    joblib.dump(model, 'trained_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("Model trained and saved successfully.")

# Evaluate the model and plot the regression line
def evaluate_model(model, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # Plotting the regression line for one feature against the target
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train['original_weight'], y_train, color='blue', label='Training Data')
    plt.scatter(X_test['original_weight'], y_test, color='green', label='Test Data')
    
    x_range = np.linspace(X_test['original_weight'].min(), X_test['original_weight'].max(), 100)
    
    # Predict using the model across the entire range
    y_range_pred = model.predict(pd.DataFrame({
        'original_weight': x_range,
        'calorie_deficit': [X_test['calorie_deficit'].mean()] * 100,
        'training_x_a_week': [X_test['training_x_a_week'].mean()] * 100,
        'average_distance': [X_test['average_distance'].mean()] * 100,
        'period_of_training_weeks': [X_test['period_of_training_weeks'].mean()] * 100
    }))
    
    # Plot the regression line
    plt.plot(x_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
    
    plt.xlabel('Original Weight')
    plt.ylabel('Weight Change')
    plt.title('Linear Regression: Original Weight vs Weight Change')
    plt.legend()

    # Save the plot as an image instead of showing it
    plt.savefig('regression_plot.png')
    print("Plot saved as 'regression_plot.png'")


# Load the model and use it for predictions
def get_user_input_and_predict():
    # Load the trained model and label encoder
    model = joblib.load('trained_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Gather user inputs
    original_weight = float(input("What's your actual weight (in kg)? "))
    calorie_deficit = input("Do you plan to reach a calorie deficit in your diet? (yes/no) ").strip().lower()
    training_x_a_week = int(input("How many days a week can you train? "))
    average_distance = float(input("What's the average distance you do in a training session (in km)? "))
    period_of_training_weeks = int(input("What's the period of training you plan to follow (in weeks)? "))

    # Process the inputs
    calorie_deficit_encoded = label_encoder.transform([calorie_deficit])[0]

    # Create a DataFrame for the model input
    user_data = pd.DataFrame({
        "original_weight": [original_weight],
        "calorie_deficit": [calorie_deficit_encoded],
        "training_x_a_week": [training_x_a_week],
        "average_distance": [average_distance],
        "period_of_training_weeks": [period_of_training_weeks]
    })

    # Predict the weight change
    predicted_weight_change = model.predict(user_data)

    # Output the prediction
    print(f"\nBased on your inputs, your predicted weight after the training period is: {predicted_weight_change[0]:.2f} kg")


train_model()

#  prediction
get_user_input_and_predict()
