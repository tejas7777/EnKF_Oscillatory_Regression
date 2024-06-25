import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import pandas as pd
from model.dnn import DNN
from model.dnn_classifier import DNN_Classifier
from index import ModelTrainer as EnKFModelTrainer
from adam_train import ModelTrainAdam

def initialize_enkf_trainer(data, target):
    model = DNN_Classifier(input_size=data.shape[1], output_size=target.shape[1] if len(target.shape) > 1 else 1)
    enkf_trainer = EnKFModelTrainer(model, lr=0.5, sigma=0.001, k=50, gamma=1e-1, max_iterations=1)
    enkf_trainer.load_data(data=data, target=target, set_standardize=False)
    return enkf_trainer

def initialize_adam_trainer(data, target):
    model = DNN_Classifier(input_size=data.shape[1], output_size=target.shape[1] if len(target.shape) > 1 else 1)
    adam_trainer = ModelTrainAdam(model)
    adam_trainer.load_data(data=data, target=target, set_standardize=False)
    return adam_trainer

def compare_validation_losses(data_path):
    # Load dataset
    data = pd.read_csv(data_path)
    X = data[[col for col in data.columns if 'Theta' in col]].values
    y = data['Label'].values.reshape(-1, 1)

    # Initialize trainers
    enkf_trainer = initialize_enkf_trainer(X, y)
    adam_trainer = initialize_adam_trainer(X, y)

    # Train models
    enkf_trainer.train(num_epochs=100)
    adam_trainer.train(num_epochs=100)

    # Plot validation losses
    plt.plot(enkf_trainer.val_loss, label='EnKF Validation Loss')
    plt.plot(adam_trainer.val_loss, label='Adam Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.title('Validation Loss Comparison')
    plt.show()

if __name__ == "__main__":
    data_path = '../dataset/simple_binary_classification_data.csv'
    compare_validation_losses(data_path)