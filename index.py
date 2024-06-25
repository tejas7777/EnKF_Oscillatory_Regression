import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optimiser.gradient_free_enkf import EnKFOptimizerGradFree
from optimiser.greadient_free_enkf_memory import EnKFOptimizerGradFreeMemory
from optimiser.enkf_classification import EnKFOptimizerClassification
from optimiser.enkf_classification_multiclass import EnKFOptimizerMultiClassification
from model.dnn import DNN
from model.dnn_classifier import DNN_Classifier
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelTrainer():
    def __init__(self,model, lr=0.5, sigma=0.001, k=50, gamma=1e-1, max_iterations=1):
        self.model = model
        #self.loss_function = nn.MSELoss()
        #self.loss_function = nn.BCELoss()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimiser = EnKFOptimizerMultiClassification(model, lr, sigma, k, gamma, max_iterations=1, debug_mode=False)

    def load_data(self, data, target, set_standardize = False, test_size=0.2, val_size=0.2):
        # Split data into training and temporary set
        X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=test_size + val_size, random_state=42)

        # Split temporary set into validation and test sets
        val_size_adjusted = val_size / (test_size + val_size)  # Adjust validation size for the reduced dataset
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        if set_standardize:
            self.standardizse_data()

        self.__convert_data_to_tensor()

    def standardizse_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def __convert_data_to_tensor(self):
        #Convert to tensors
        # self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        # self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        # self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        # self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        # self.y_val = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1)
        # self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)  # Convert to long for CrossEntropyLoss
        self.y_val = torch.tensor(self.y_val, dtype=torch.long)      # Convert to long for CrossEntropyLoss
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)

    def F(self, parameters):
        with torch.no_grad(): 
            for original_param, new_param in zip(self.model.parameters(), parameters):
                #original_param.grad = None
                original_param.data.copy_(new_param.data)

            #Perform the forward pass with the adjusted parameters
            output =  self.model(self.X_train)

            return output


    # def train(self, num_epochs=100, is_plot_graph = 0):
    #     train_losses = []
    #     val_losses = []

    #     print("TRAINING STARTED ...")
    #     for epoch in range(num_epochs):
    #         self.optimiser.step(F=self.F, obs=self.y_train)

    #         #self.model.eval()
    #         with torch.no_grad():
    #             train_output = self.model(self.X_train)
    #             train_loss = self.loss_function(train_output, self.y_train)
    #             val_output = self.model(self.X_val)
    #             val_loss = self.loss_function(val_output, self.y_val)
    #             train_losses.append(train_loss.item())
    #             val_losses.append(val_loss.item())

    #         print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

    #     if is_plot_graph:
    #         self.plot_train_graph(train_losses, val_losses)

    #     self.train_loss = train_losses
    #     self.val_loss = val_losses

    def train(self, num_epochs=500, is_plot_graph = 0):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        print("TRAINING STARTED ...")
        for epoch in range(num_epochs):
            self.optimiser.step(F=self.F, obs=self.y_train)

            #self.model.eval()
            with torch.no_grad():
                # Calculate train loss
                train_output = self.model(self.X_train)
                train_loss = self.loss_function(train_output, self.y_train)
                train_losses.append(train_loss.item())

                # Calculate validation loss
                val_output = self.model(self.X_val)
                val_loss = self.loss_function(val_output, self.y_val)
                val_losses.append(val_loss.item())

                # Calculate train accuracy
                _, train_predicted = torch.max(train_output, 1)
                train_correct = (train_predicted == self.y_train).sum().item()
                train_accuracy = train_correct / self.y_train.size(0)
                train_accuracies.append(train_accuracy)

                # Calculate validation accuracy
                _, val_predicted = torch.max(val_output, 1)
                val_correct = (val_predicted == self.y_val).sum().item()
                val_accuracy = val_correct / self.y_val.size(0)
                val_accuracies.append(val_accuracy)

                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}, Train Accuracy: {train_accuracy*100:.2f}%, Val Accuracy: {val_accuracy*100:.2f}%')

        if is_plot_graph:
            self.plot_train_graph(train_losses, val_losses)

        self.train_loss = train_losses
        self.val_loss = val_losses
        self.train_accuracy = train_accuracies
        self.val_accuracy = val_accuracies

    def plot_train_graph(self,train_losses, val_losses):
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

        #print(f'Test Loss: {test_loss.item()}')
    
    def save_model(self, filename=None):
        if filename is None:
            filename = f'model_enkf.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model, save_path)
        print(f'Complete model saved to {save_path}')

    def get_ensemble_particles(self):
        return [self.optimiser.unflatten_parameters(particle) for particle in self.optimiser.particles.T]
    
    def plot_ensemble_particles_distribution(self):
        particles = self.get_ensemble_particles()
        flattened_particles = [np.concatenate([p.detach().cpu().numpy().flatten() for p in particle]) for particle in particles]
        
        num_particles = len(flattened_particles)
        fig = make_subplots(rows=num_particles, cols=1, subplot_titles=[f'Particle {i+1} Distribution' for i in range(num_particles)])
        
        for i, particle in enumerate(flattened_particles):
            fig.add_trace(
                go.Histogram(x=particle, nbinsx=50, name=f'Particle {i+1}'),
                row=i+1, col=1
            )
        
        fig.update_layout(height=300 * num_particles, width=800, title_text="Particle Distributions", showlegend=False)
        fig.show()
    

# #Dataset
# data = pd.read_csv('./dataset/oscillatory_data_large.csv')
# X = data[[col for col in data.columns if 'Theta' in col]].values
# y = data[[col for col in data.columns if 'F_Theta' in col]].values

# # # data = pd.read_csv('./dataset/complex_regression_data.csv')
# # # X = data[[col for col in data.columns if col.startswith('Feature_')]].values
# # # y = data['Target'].values.reshape(-1, 1)

# # Load the complex binary classification data
data = pd.read_csv('./dataset/multi_class_classification_data.csv')

#data = pd.read_csv('./dataset/simple_binary_classification_data.csv')

# Extract the feature matrix X and the target vector y
X = data[[col for col in data.columns if 'Theta' in col]].values
#y = data['Label'].values.reshape(-1, 1)
y = data['Label'].values

print (y.shape)


model_train = ModelTrainer(model=DNN_Classifier(input_size=X.shape[1], output_size=5))
model_train.load_data(data=X, target=y)
model_train.train(is_plot_graph=1)
model_train.evaluate()
model_train.save_model('model_enkf.pth')