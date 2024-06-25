import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from model.dnn_classifier import DNN_Classifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from index import ModelTrainer  # Make sure this is the EnKF model trainer for multi-class classification

# Load dataset
data = pd.read_csv('../dataset/multi_class_classification_data.csv')
X = data[[col for col in data.columns if 'Theta' in col]].values
y = data['Label'].values  # Ensure labels are integers for classification

# Function to run experiments with different hyperparameters
def run_experiments(hyperparameters_list, X, y):
    results = []
    for params in hyperparameters_list:
        lr, sigma, k, gamma = params
        model = DNN_Classifier(input_size=X.shape[1], output_size=len(set(y)))
        model_trainer = ModelTrainer(model, lr, sigma, k, gamma, max_iterations=1)
        model_trainer.load_data(data=X, target=y)
        model_trainer.train(num_epochs=100)
        model_trainer.evaluate()

        results.append({
            'lr': lr,
            'sigma': sigma,
            'k': k,
            'gamma': gamma,
            'train_loss': model_trainer.train_loss,
            'val_loss': model_trainer.val_loss,
            'train_accuracy': model_trainer.train_accuracy,
            'val_accuracy': model_trainer.val_accuracy,
            'model_parameters': [p.data.cpu().numpy() for p in model_trainer.model.parameters()]
        })

        model_trainer.plot_ensemble_particles_distribution()

    return results

def plot_parameter_distributions(results):
    fig, axs = plt.subplots(len(results), 1, figsize=(10, 5 * len(results)))

    if len(results) == 1:
        axs = [axs]  # Ensure axs is iterable

    for i, result in enumerate(results):
        params_flattened = np.concatenate([param.flatten() for param in result['model_parameters']])
        axs[i].hist(params_flattened, bins=50, alpha=0.7, label=f"lr={result['lr']}, sigma={result['sigma']}, k={result['k']}, gamma={result['gamma']}")
        axs[i].legend()
        axs[i].set_title(f"Parameter Distribution for lr={result['lr']}, sigma={result['sigma']}, k={result['k']}, gamma={result['gamma']}")
        axs[i].set_xlabel('Parameter Value')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Define hyperparameters to test
hyperparameters_list = [
    (0.5, 0.01, 50, 1e-1),
]

# Run experiments
results = run_experiments(hyperparameters_list, X, y)

# Plot results
plot_parameter_distributions(results)
