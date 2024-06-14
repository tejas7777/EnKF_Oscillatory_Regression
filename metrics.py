import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from model.dnn import DNN  # Ensure this path is correct based on your directory structure

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():  # Ensure no gradients are calculated
        predictions = model(X_test)
        test_loss = nn.MSELoss()(predictions, y_test)
        print(f'Test Loss (MSE): {test_loss.item()}')

        # Additional metrics
        y_test_np = y_test.numpy()
        predictions_np = predictions.numpy()

        mae = mean_absolute_error(y_test_np, predictions_np)
        r2 = r2_score(y_test_np, predictions_np)
        mse = mean_squared_error(y_test_np, predictions_np)

        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R^2 Score: {r2}')
        print(f'Mean Squared Error (MSE): {mse}')

# Load your data
data = pd.read_csv('oscillatory_data_large.csv')
X = data[[col for col in data.columns if 'Theta' in col]].values
y = data[[col for col in data.columns if 'F_Theta' in col]].values

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Load the entire saved model
model = torch.load('./saved_models/model_enkf.pth')

# Evaluate the model
evaluate_model(model, X_test, y_test)