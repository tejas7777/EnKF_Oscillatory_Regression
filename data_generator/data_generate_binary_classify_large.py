import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Parameters
n_samples = 1000  # Number of samples
n_features = 50    # Number of features in each theta vector
noise_std = 0.1     # Standard deviation for Gaussian noise
threshold = 0.5     # Threshold for binary classification

# Generate random feature vectors theta of size 10000x200 (each row is a sample)
theta = np.random.randn(n_samples, n_features)

# Generate random weights for different transformations
W1 = np.random.randn(n_features, n_features)
W2 = np.random.randn(n_features, n_features)
W3 = np.random.randn(n_features, 1)

# Generate non-linear transformations of the input features
Z1 = np.tanh(np.dot(theta, W1))
Z2 = np.tanh(np.dot(Z1, W2))
linear_combination = np.dot(Z2, W3)

# Add Gaussian noise
noisy_linear_combination = linear_combination + np.random.normal(0, noise_std, linear_combination.shape)

# Apply sigmoid to get probabilities
probabilities = 1 / (1 + np.exp(-noisy_linear_combination))

# Apply threshold to get binary labels
labels = (probabilities > threshold).astype(int).flatten()

# Create a DataFrame
theta_df = pd.DataFrame(theta, columns=[f'Theta_{i+1}' for i in range(n_features)])
labels_df = pd.DataFrame(labels, columns=['Label'])

# Concatenate theta and labels into one DataFrame
data = pd.concat([theta_df, labels_df], axis=1)

# Save the data
data.to_csv('../dataset/complex_binary_classification_data.csv', index=False)

# # Save the true parameters
# true_params = {'W1': W1, 'W2': W2, 'W3': W3}
# np.save('true_params_complex_binary_classification.npy', true_params)

print("Complex binary classification data")