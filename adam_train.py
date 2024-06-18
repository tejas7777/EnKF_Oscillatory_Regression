import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.dnn import DNN
import pandas as pd

class ModelTrainAdam:
    
    def __init__(self, model):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-1)

    def load_data(self, data, target, set_standardize=False, test_size=0.2, val_size=0.2):
        X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=test_size + val_size, random_state=42)

         # Split temporary set into validation and test sets
        val_size_adjusted = val_size / (test_size + val_size)  # Adjust validation size for the reduced dataset
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        if set_standardize:
            self.standardize_data()

        self.__convert_data_to_tensor()

    def standardize_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def __convert_data_to_tensor(self):
        # Convert to tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

    def train(self, num_epochs=100):
        print("TRAINING STARTED ...")

        train_losses = []
        val_losses = []


        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(self.X_train)
            loss = self.loss_function(output, self.y_train)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(self.X_val)
                val_loss = self.loss_function(val_output, self.y_val)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()} Val Loss: {val_loss.item()}')

        self.val_loss = val_losses

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(self.X_test)
            test_loss = self.loss_function(test_output, self.y_test)
        print(f'Test Loss: {test_loss.item()}')

# Dataset
# data = pd.read_csv('./dataset/oscillatory_data_large.csv')
# X = data[[col for col in data.columns if 'Theta' in col]].values
# y = data[[col for col in data.columns if 'F_Theta' in col]].values

# # data = pd.read_csv('./dataset/complex_regression_data.csv')
# # X = data[[col for col in data.columns if col.startswith('Feature_')]].values
# # y = data['Target'].values.reshape(-1, 1)

# model_train_adam = ModelTrainAdam(model=DNN(input_size=X.shape[1], output_size=y.shape[1] if len(y.shape) > 1 else 1 ))
# model_train_adam.load_data(data=X, target=y)
# model_train_adam.train(num_epochs=100)
# model_train_adam.evaluate()