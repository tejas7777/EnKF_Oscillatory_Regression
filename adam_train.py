import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.dnn import DNN
from model.dnn_classifier import DNN_Classifier
import pandas as pd
import matplotlib.pyplot as plt

# class ModelTrainAdam:
    
#     def __init__(self, model):
#         self.model = model
#         #self.loss_function = nn.MSELoss()
#         self.loss_function = nn.BCELoss()
#         self.optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     def load_data(self, data, target, set_standardize=False, test_size=0.2, val_size=0.2):
#         X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=test_size + val_size, random_state=42)

#          # Split temporary set into validation and test sets
#         val_size_adjusted = val_size / (test_size + val_size)  # Adjust validation size for the reduced dataset
#         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

#         self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
#         self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

#         if set_standardize:
#             self.standardize_data()

#         self.__convert_data_to_tensor()

#     def standardize_data(self):
#         scaler = StandardScaler()
#         self.X_train = scaler.fit_transform(self.X_train)
#         self.X_test = scaler.transform(self.X_test)

#     def __convert_data_to_tensor(self):
#         # Convert to tensors
#         self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
#         self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
#         self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
#         self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
#         self.y_val = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1)
#         self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

#     def train(self, num_epochs=100, is_plot_graph = 0):
#         print("TRAINING STARTED ...")

#         train_losses = []
#         val_losses = []


#         for epoch in range(num_epochs):
#             self.model.train()
#             self.optimizer.zero_grad()
            
#             # Forward pass
#             output = self.model(self.X_train)
#             loss = self.loss_function(output, self.y_train)
            
#             # Backward pass and optimization
#             loss.backward()
#             self.optimizer.step()
#             self.model.eval()
#             with torch.no_grad():
#                 val_output = self.model(self.X_val)
#                 val_loss = self.loss_function(val_output, self.y_val)

#             train_losses.append(loss.item())
#             val_losses.append(val_loss.item())
            
#             print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()} Val Loss: {val_loss.item()}')

#         if is_plot_graph:
#             self.plot_train_graph(train_losses, val_losses)

#         self.val_loss = val_losses

#     def plot_train_graph(self,train_losses, val_losses):
#         # Plot training and validation loss
#         plt.plot(train_losses, label='Train Loss')
#         plt.plot(val_losses, label='Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.show()

#     def evaluate(self):
#         self.model.eval()
#         with torch.no_grad():
#             test_output = self.model(self.X_test)
#             test_loss = self.loss_function(test_output, self.y_test)
#         print(f'Test Loss: {test_loss.item()}')

# # Dataset
# # data = pd.read_csv('./dataset/oscillatory_data_large.csv')

# # data = pd.read_csv('./dataset/simple_binary_classification_data.csv')
# # X = data[[col for col in data.columns if 'Theta' in col]].values
# # y = data['Label'].values.reshape(-1, 1)

# # # # data = pd.read_csv('./dataset/complex_regression_data.csv')
# # # # X = data[[col for col in data.columns if col.startswith('Feature_')]].values
# # # # y = data['Target'].values.reshape(-1, 1)

# # model_train_adam = ModelTrainAdam(model=DNN_Classifier(input_size=X.shape[1], output_size=y.shape[1] if len(y.shape) > 1 else 1 ))
# # model_train_adam.load_data(data=X, target=y)
# # model_train_adam.train(num_epochs=500, is_plot_graph=1)
# # model_train_adam.evaluate()

class ModelTrainAdam:
    
    def __init__(self, model):
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)  # Convert to long for CrossEntropyLoss
        self.y_val = torch.tensor(self.y_val, dtype=torch.long)      # Convert to long for CrossEntropyLoss
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)    # Convert to long for CrossEntropyLoss

    def train(self, num_epochs=100, is_plot_graph = 0):
        print("TRAINING STARTED ...")

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(self.X_train)
            loss = self.loss_function(output, self.y_train)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Calculate train accuracy
            _, train_predicted = torch.max(output, 1)
            train_correct = (train_predicted == self.y_train).sum().item()
            train_accuracy = train_correct / self.y_train.size(0)
            train_accuracies.append(train_accuracy)

            self.model.eval()
            with torch.no_grad():
                val_output = self.model(self.X_val)
                val_loss = self.loss_function(val_output, self.y_val)

                # Calculate validation accuracy
                _, val_predicted = torch.max(val_output, 1)
                val_correct = (val_predicted == self.y_val).sum().item()
                val_accuracy = val_correct / self.y_val.size(0)
                val_accuracies.append(val_accuracy)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}, Train Accuracy: {train_accuracy*100:.2f}%, Val Accuracy: {val_accuracy*100:.2f}%')

        if is_plot_graph:
            self.plot_train_graph(train_losses, val_losses)

        self.train_loss = train_losses
        self.val_loss = val_losses
        self.train_accuracy = train_accuracies
        self.val_accuracy = val_accuracies

    def plot_train_graph(self, train_losses, val_losses):
        # Plot training and validation loss
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(self.X_test)
            test_loss = self.loss_function(test_output, self.y_test)

            # Calculate test accuracy
            _, test_predicted = torch.max(test_output, 1)
            test_correct = (test_predicted == self.y_test).sum().item()
            test_accuracy = test_correct / self.y_test.size(0)

        print(f'Test Loss: {test_loss.item()}, Test Accuracy: {test_accuracy*100:.2f}%')


# Load the multi-class classification data
data = pd.read_csv('./dataset/multi_class_classification_data.csv')

# Extract the feature matrix X and the target vector y
X = data[[col for col in data.columns if 'Theta' in col]].values
y = data['Label'].values  # Keep labels as integer class indices

# Initialize the model trainer with appropriate model and output size
model_train_adam = ModelTrainAdam(model=DNN_Classifier(input_size=X.shape[1], output_size=5 ))

# Load the data into the model trainer
model_train_adam.load_data(data=X, target=y)
model_train_adam.train(num_epochs=500, is_plot_graph=1)
model_train_adam.evaluate()
