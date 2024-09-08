import pandas as pd
import joblib

# Load the model and label encoder
def load_model_and_encoder():
    knn_model = joblib.load('knn_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return knn_model, label_encoder

# Predict based on user input
def get_user_input_and_predict(knn_model, label_encoder):
    original_weight = float(input("What's your actual weight (in kg)? "))
    calorie_deficit = input("Do you plan to reach a calorie deficit in your diet? (yes/no) ").strip().lower()
    training_x_a_week = int(input("How many days a week can you train? "))
    average_distance = float(input("What's the average distance you do in a training session (in km)? "))
    period_of_training_weeks = int(input("What's the period of training you plan to follow (in weeks)? "))

    calorie_deficit_encoded = label_encoder.transform([calorie_deficit])[0]

    user_data = pd.DataFrame({
        "original_weight": [original_weight],
        "calorie_deficit": [calorie_deficit_encoded],
        "training_x_a_week": [training_x_a_week],
        "average_distance": [average_distance],
        "period_of_training_weeks": [period_of_training_weeks]
    })

    predicted_weight_change = knn_model.predict(user_data)
    print(f"\nBased on your inputs, your predicted weight after the training period is: {predicted_weight_change[0]:.2f} kg")

# Main function to load the model and make predictions
def main():
    knn_model, label_encoder = load_model_and_encoder()
    get_user_input_and_predict(knn_model, label_encoder)

if __name__ == "__main__":
    main()
