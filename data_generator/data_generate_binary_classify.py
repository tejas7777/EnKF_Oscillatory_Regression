import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Parameters
n_samples = 10000  # Number of samples
n_features = 200   # Number of features
n_informative = 150  # Number of informative features
n_redundant = 25   # Number of redundant features
n_clusters_per_class = 2  # Number of clusters per class
class_sep = 5.0    # Separation between classes
random_state = 450  # Seed for reproducibility

# Generate the dataset
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class,
                           class_sep=class_sep, random_state=random_state)

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=[f'Theta_{i+1}' for i in range(n_features)])
y_df = pd.DataFrame(y, columns=['Label'])

# Concatenate features and labels into one DataFrame
data = pd.concat([X_df, y_df], axis=1)

# Save the data
data.to_csv('../dataset/simple_binary_classification_data.csv', index=False)

print("Simple binary classification data saved.")
