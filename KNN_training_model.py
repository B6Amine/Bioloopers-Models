import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#  Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    
    # Encode the 'calorie_deficit' column
    le = LabelEncoder()
    df['calorie_deficit'] = le.fit_transform(df['calorie_deficit'])
    
    # Define features (X) and target (y)
    X = df[['original_weight', 'calorie_deficit', 'training_x_a_week', 'average_distance', 'period_of_training_weeks']]
    y = df['weight_change']
    
    return X, y, le

#  Train the KNN model
def train_knn_model(X_train, y_train, n_neighbors=5):
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")

# Main function to train, evaluate, and save the model
def main():
    file_path = 'C:/Users/Amine/Desktop/Training_Data_1000.xlsx' 
    X, y, label_encoder = load_and_preprocess_data(file_path)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the KNN model
    knn_model = train_knn_model(X_train, y_train, n_neighbors=5)
    
    # Evaluate the model
    print("Evaluating the model on the test data:")
    evaluate_model(knn_model, X_test, y_test)
    
    # Save the model and label encoder
    joblib.dump(knn_model, 'knn_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model and label encoder have been saved.")

if __name__ == "__main__":
    main()
