import torch
import pandas as pd
from model.dnn_classifier import DNN_Classifier
import torch.nn as nn

# Function to load testing data
def load_testing_data(filepath):
    data = pd.read_csv(filepath)
    X_test = data[[col for col in data.columns if 'Theta' in col]].values
    y_test = data['Label'].values
    return X_test, y_test

# Function to load the model
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Set model to evaluation mode
    return model

# Load the testing data
X_test, y_test = load_testing_data('./dataset/multi_class_classification_data_test.csv')

# Convert to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Load the model
model = load_model('./saved_models/model_enkf.pth')

# Define the loss function
loss_function = nn.CrossEntropyLoss()

# Evaluate the model
with torch.no_grad():
    test_output = model(X_test_tensor)
    test_loss = loss_function(test_output, y_test_tensor)

    # Calculate test accuracy
    _, test_predicted = torch.max(test_output, 1)
    test_correct = (test_predicted == y_test_tensor).sum().item()
    test_accuracy = test_correct / y_test_tensor.size(0)

print(f'Test Loss: {test_loss.item()}, Test Accuracy: {test_accuracy*100:.2f}%')
