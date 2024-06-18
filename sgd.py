import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.dnn import DNN
import pandas as pd
import numpy as np

class ModelTrainSGD:
    
    def __init__(self, model):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=1e-2)

    def load_data(self, data, target, set_standardize=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.2, random_state=42)

        if set_standardize:
            self.standardize_data()

        self.__convert_data_to_tensor()

    def standardize_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def __convert_data_to_tensor(self):
        #first checking if pandas type
        self.X_train = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        self.X_test = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
        self.y_train = self.y_train.values if isinstance(self.y_train, pd.DataFrame) else self.y_train

        # Convert to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

    def train(self, num_epochs=10):
        print("TRAINING STARTED ...")
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(self.X_train)
            loss = self.loss_function(output, self.y_train)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(self.X_test)
            test_loss = self.loss_function(test_output, self.y_test)
        print(f'Test Loss: {test_loss.item()}')

#Dataset



data = pd.read_csv('./dataset/complex_regression_data.csv')
X = data[[col for col in data.columns if col.startswith('Feature_')]].values
y = data['Target'].values.reshape(-1, 1)
# X = data[[col for col in data.columns if 'Theta' in col]].values
# y = data[[col for col in data.columns if 'F_Theta' in col]].values

model_train_sgd = ModelTrainSGD(model=DNN(input_size=X.shape[1], output_size=y.shape[1] if len(y.shape) > 1 else 1))
model_train_sgd.load_data(data=X, target=y)
model_train_sgd.train(num_epochs=100)
model_train_sgd.evaluate()

