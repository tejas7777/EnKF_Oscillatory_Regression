import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from model.dnn import DNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from index import ModelTrainer

# Load dataset
data = pd.read_csv('../dataset/oscillatory_data_large.csv')
X = data[[col for col in data.columns if 'Theta' in col]].values
y = data[[col for col in data.columns if 'F_Theta' in col]].values

true_params = np.load('../dataset/true_params.npy', allow_pickle=True).item()

# Function to run experiments with different hyperparameters
def run_experiments(hyperparameters_list, X, y):
    results = []
    for params in hyperparameters_list:
        lr, sigma, k, gamma = params
        model = DNN(input_size=X.shape[1], output_size=y.shape[1] if len(y.shape) > 1 else 1)
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
            'model_parameters': [p.data.cpu().numpy() for p in model_trainer.model.parameters()]
        })

    #analyze_first_layer_transformation(model, true_params, X)

        model_trainer.plot_ensemble_particles_distribution()

    return results

def plot_parameter_distributions(results, true_params):
    fig, axs = plt.subplots(len(results) + 1, 1, figsize=(10, 5 * (len(results) + 1)))

    # Ensure axs is always an array of Axes objects
    if len(results) == 0:
        axs = np.array([axs])
    elif len(results) == 1:
        axs = np.array([axs]).flatten()

    # Plot true parameters distribution
    true_params_flattened = np.concatenate([param.flatten() for param in true_params.values()])
    axs[0].hist(true_params_flattened, bins=50, alpha=0.7, color='g', label='True Parameters')
    axs[0].legend()
    axs[0].set_title('True Parameter Distribution')
    axs[0].set_xlabel('Parameter Value')
    axs[0].set_ylabel('Frequency')

    # Plot learned parameters distribution for each hyperparameter setting
    for i, result in enumerate(results):
        params_flattened = np.concatenate([param.flatten() for param in result['model_parameters']])
        axs[i + 1].hist(params_flattened, bins=50, alpha=0.7, label=f"lr={result['lr']}, sigma={result['sigma']}, k={result['k']}, gamma={result['gamma']}")
        axs[i + 1].legend()
        axs[i + 1].set_title(f"Parameter Distribution for lr={result['lr']}, sigma={result['sigma']}, k={result['k']}, gamma={result['gamma']}")
        axs[i + 1].set_xlabel('Parameter Value')
        axs[i + 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def analyze_first_layer_transformation(model, true_params_dict, X):
    # Assuming the first layer is a linear layer
    first_layer_weights = list(model.parameters())[0].data.numpy()
    
    # Extract true coefficients from the true_params_dict
    true_coefficients = np.concatenate([param.flatten() for param in true_params_dict.values()])
    
    # Visualize the weights
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(first_layer_weights.flatten(), bins=50, alpha=0.7, color='b', label='First Layer Weights')
    plt.title("First Layer Weights Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(true_coefficients, bins=50, alpha=0.7, color='g', label='True Coefficients')
    plt.title("True Coefficients Distribution")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Apply the transformations
    transformed_features_model = np.dot(X, first_layer_weights)
    transformed_features_true = np.dot(X, true_coefficients.reshape(-1, 1))

    # Visualize the transformed features
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(transformed_features_model.flatten(), bins=50, alpha=0.7, color='b', label='Transformed Features (Model)')
    plt.title("Transformed Features by First Layer Weights")
    plt.xlabel("Transformed Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(transformed_features_true.flatten(), bins=50, alpha=0.7, color='g', label='Transformed Features (True)')
    plt.title("Transformed Features by True Coefficients")
    plt.xlabel("Transformed Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()
# Define hyperparameters to test
hyperparameters_list = [
    # (1e-2, 0.001, 500, 1e-1),
    # (1e-1, 0.001, 500, 1e-1),
    (0.5, 0.01, 50, 1e-1),
    # (1e-1, 0.001, 500, 1e-1),
    # (1e-1, 0.01, 500, 1e-1),
    #(1e-1, 0.1, 50, 1e-1),
    # (1e-1, 0.001, 500, 1e-2),
    # (1e-1, 0.001, 500, 1e-3),
    # (1e-1, 0.001, 500, 1e-4),
]



# Run experiments
results = run_experiments(hyperparameters_list, X, y)

# Plot results
# for result in results:
#     #plt.plot(result['train_loss'], label=f"Train Loss (lr={result['lr']}, sigma={result['sigma']}, k={result['k']}, gamma={result['gamma']})")
#     #plt.plot(result['val_loss'], label=f"Val Loss (lr={result['lr']}, sigma={result['sigma']}, gamma={result['gamma']})")
#     plot_parameter_distributions(results, true_params=true_params)

# plt.title('Validation Loss (K=500)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
