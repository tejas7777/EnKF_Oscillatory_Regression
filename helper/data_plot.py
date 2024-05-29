import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../oscillatory_data_small.csv'  # Replace with the correct path to your file
data = pd.read_csv(file_path)

# Extract the output F(theta)
output_column = 'F_Theta_1'

# Plot the output column
plt.figure(figsize=(14, 7))
plt.plot(data[output_column], label=output_column)

plt.xlabel('Sample Index')
plt.ylabel('F(Theta)')
plt.title('Oscillatory Regression Function: F(Theta)')
plt.legend()
plt.grid(True)
plt.show()