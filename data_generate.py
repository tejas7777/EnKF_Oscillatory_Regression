import numpy as np
import pandas as pd

# Parameters
n_samples = 10000  # Number of samples
n_features = 20   # Number of features in each theta vector
m_output = 1     # Dimension of the output vector
epsilon = 1.0      # Scale factor for the sinusoidal component
l = 20.0           # Frequency multiplier for the sinusoidal component
noise_std = 0.1    # Standard deviation for Gaussian noise

# Generate random matrices A and B of size 300x200
A = np.random.randn(m_output, n_features)
B = np.random.randn(m_output, n_features)

# Generate random feature vectors theta of size 10000x200 (each row is a sample)
theta = np.random.randn(n_samples, n_features)

# Initialize the output array
F_theta = np.zeros((n_samples, m_output))

# Calculate the function F(theta) for each sample
for i in range(n_samples):
    A_theta = np.dot(A, theta[i])
    B_theta = np.dot(B, theta[i])
    sin_component = epsilon * np.sin(l * B_theta)
    F_theta[i] = A_theta + sin_component

# Add Gaussian noise to F_theta
noisy_F_theta = F_theta + np.random.normal(0, noise_std, F_theta.shape)

# Create a DataFrame
theta_df = pd.DataFrame(theta, columns=[f'Theta_{i+1}' for i in range(n_features)])
F_theta_df = pd.DataFrame(noisy_F_theta, columns=[f'F_Theta_{j+1}' for j in range(m_output)])

# Concatenate theta and F_theta into one DataFrame
data = pd.concat([theta_df, F_theta_df], axis=1)

# Save the DataFrame to a CSV file
data.to_csv('oscillatory_data_small.csv', index=False)