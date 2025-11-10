import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the epsilon value you want to compare
EPSILON = 0.10  # Change this value to test other epsilons

# File paths
DATASET = "MNIST"
root_path = "/local/kat/LESLIE/Topic_Dimshield"
original_csv = f'{root_path}/FGSM/verify/confidence_results_original.csv'
eps_csv = f'{root_path}/FGSM/verify/confidence_results_eps_{EPSILON:.2f}.csv'
plot_path = f'{root_path}/FGSM/verify/confidence_comparison_eps_{EPSILON:.2f}.png'

# Load the data
original = pd.read_csv(original_csv)
eps_df = pd.read_csv(eps_csv)

# Get the highest confidence for each index (row)
original_max = original.max(axis=1)
eps_max = eps_df.max(axis=1)

# Compute average lines
original_avg = np.mean(original_max)
eps_avg = np.mean(eps_max)

# Plot
plt.figure(figsize=(12, 6))
indices = np.arange(len(original_max))

plt.plot(indices, original_max, label='Original', alpha=0.7)
plt.plot(indices, eps_max, label=f'Eps {EPSILON:.2f}', alpha=0.7)

# Plot average lines
plt.hlines(original_avg, xmin=0, xmax=len(original_max)-1, colors='blue', linestyles='dashed', label='Original Avg')
plt.hlines(eps_avg, xmin=0, xmax=len(eps_max)-1, colors='orange', linestyles='dashed', label=f'Eps {EPSILON:.2f} Avg')

plt.xlabel('Index')
plt.ylabel('Highest Confidence')
plt.title(f'Spread of Highest Confidence per Index: Original vs Eps {EPSILON:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Plot saved to {plot_path}")