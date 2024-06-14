import numpy as np
import pandas as pd

# Parameters
n_samples = 10000  # Number of samples
n_features = 20   #Number of features in each theta vector
m_output = 1     #Dimension of the output vector
epsilon = 1.0  
l = 20.0           #sinusoidal component
noise_std = 0.1    #Standard deviation

A = np.random.randn(m_output, n_features)
B = np.random.randn(m_output, n_features)

theta = np.random.randn(n_samples, n_features)

F_theta = np.zeros((n_samples, m_output))

for i in range(n_samples):
    A_theta = np.dot(A, theta[i])
    B_theta = np.dot(B, theta[i])
    sin_component = epsilon * np.sin(l * B_theta)
    F_theta[i] = A_theta + sin_component

# Add noise to F_theta (which is the final output column)
noisy_F_theta = F_theta + np.random.normal(0, noise_std, F_theta.shape)

theta_df = pd.DataFrame(theta, columns=[f'Theta_{i+1}' for i in range(n_features)])
F_theta_df = pd.DataFrame(noisy_F_theta, columns=[f'F_Theta_{j+1}' for j in range(m_output)])

data = pd.concat([theta_df, F_theta_df], axis=1)
data.to_csv('oscillatory_data_small.csv', index=False)