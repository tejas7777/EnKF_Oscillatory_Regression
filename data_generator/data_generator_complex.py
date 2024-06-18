import numpy as np
import pandas as pd

# Parameters
n_samples = 5000  # Number of samples
n_features = 200  # Number of features

# Random seed for reproducibility
np.random.seed(42)

# Generate random feature matrix (input variables)
X = np.random.randn(n_samples, n_features)

# Initialize coefficients for a polynomial relationship
linear_coeffs = np.random.rand(n_features) * 2 - 1  # Coefficients between -1 and 1
quad_coeffs = np.random.rand(n_features) * 1 - 0.5  # Coefficients between -0.5 and 0.5
expo_coeffs = np.random.rand(n_features) * 0.5  # Coefficients for exponential scale
interaction_coeffs = np.random.rand(n_features, n_features) * 0.1  # Coefficients for interaction terms

# Calculate the target variable using a mix of linear, quadratic, exponential, and interaction transformations
Y_linear = np.dot(X, linear_coeffs)
Y_quad = np.dot(X**2, quad_coeffs)
Y_expo = np.dot(np.exp(X), expo_coeffs)

# Initialize interaction term to zero
Y_interaction = np.zeros(n_samples)

# Calculate interaction term
for i in range(n_features):
    for j in range(n_features):
        Y_interaction += interaction_coeffs[i, j] * X[:, i] * X[:, j]

# Add all components
Y = Y_linear + Y_quad + Y_expo + Y_interaction

# Add noise to the target variable
noise = np.random.normal(0, 0.1, Y.shape)
Y_noisy = Y + noise

# Convert to DataFrame
feature_columns = [f'Feature_{i+1}' for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_columns)
Y_df = pd.DataFrame(Y_noisy, columns=['Target'])

# Concatenate X and Y into one DataFrame
data = pd.concat([X_df, Y_df], axis=1)

# Save to CSV
data.to_csv('../dataset/complex_regression_data.csv', index=False)

print("Data generated and saved to complex_regression_data.csv")
