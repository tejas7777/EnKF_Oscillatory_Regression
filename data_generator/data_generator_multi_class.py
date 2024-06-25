import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Parameters
n_samples = 1000  # Number of samples
n_features = 20    # Number of features
n_informative = 12  # Number of informative features
n_redundant = 2    # Number of redundant features
n_clusters_per_class = 8  # Number of clusters per class
n_classes = 5      # Number of classes
class_sep = 4    # Separation between classes
random_state = 42  # Seed for reproducibility

# Generate the dataset
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class,
                           n_classes=n_classes, class_sep=class_sep, random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DataFrame
X_train_df = pd.DataFrame(X_train, columns=[f'Theta_{i+1}' for i in range(n_features)])
y_train_df = pd.DataFrame(y_train, columns=['Label'])

X_test_df = pd.DataFrame(X_test, columns=[f'Theta_{i+1}' for i in range(n_features)])
y_test_df = pd.DataFrame(y_test, columns=['Label'])

# Concatenate features and labels into one DataFrame
train_data = pd.concat([X_train_df, y_train_df], axis=1)
test_data = pd.concat([X_test_df, y_test_df], axis=1)

# Save the training data
train_data.to_csv('../dataset/multi_class_classification_data_train.csv', index=False)

# Save the testing data
test_data.to_csv('../dataset/multi_class_classification_data_test.csv', index=False)

print("Training and testing data saved.")