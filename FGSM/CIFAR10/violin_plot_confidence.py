import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os

# Directory containing the confidence CSV files
DATASET = "CIFAR10"
root_path = "/local/kat/LESLIE/Topic_Dimshield"
conf_dir = f'{root_path}/FGSM/{DATASET}/Conf_Result'
original_csv = os.path.join(conf_dir, 'confidence_results_original.csv')
plot_path = os.path.join(conf_dir, 'violin_plot_confidence_comparison_selected_eps.png')

# ---- Specify the epsilon values you want to plot ----
selected_epsilons = [0.020, 0.040, 0.060, 0.080, 0.100]  # Edit this list as needed

# Find all confidence_results_eps_*.csv files
eps_files = glob.glob(os.path.join(conf_dir, 'confidence_results_eps_*.csv'))

# Extract eps values and sort files by eps
eps_pattern = re.compile(r'confidence_results_eps_([0-9.]+)\.csv')
eps_file_tuples = []
for f in eps_files:
    match = eps_pattern.search(os.path.basename(f))
    if match:
        eps_val = float(match.group(1))
        if np.any(np.isclose(eps_val, selected_epsilons, atol=1e-6)):
            eps_file_tuples.append((eps_val, f))
eps_file_tuples.sort()  # sort by eps value

# Load original data
original = pd.read_csv(original_csv)
conf_cols = [col for col in original.columns if col.startswith('Conf_')]
original_max = original[conf_cols].max(axis=1) * 100

# Prepare data for violin plot
data = [original_max]
labels = ['Original']
avg_lines = [np.mean(original_max)]

# Load each selected eps file and extract max confidence
for eps_val, fname in eps_file_tuples:
    eps_df = pd.read_csv(fname)
    eps_max = eps_df[conf_cols].max(axis=1) * 100
    data.append(eps_max)
    labels.append(f"Eps {eps_val:.2f}")
    avg_lines.append(np.mean(eps_max))

# Plot
plt.figure(figsize=(2 + 2*len(labels), 7))
parts = plt.violinplot(data, showmeans=True, showmedians=True)

plt.xticks(np.arange(1, len(labels)+1), labels, rotation=30)
plt.ylabel('Highest Confidence (%)')
plt.title('Violin Plot of Highest Confidence: Original vs Selected Epsilons')

# Plot average lines for each group
for i, avg in enumerate(avg_lines):
    plt.hlines(avg, xmin=i+0.85, xmax=i+1.15, colors='C'+str(i), linestyles='dashed', label=f'{labels[i]} Avg')

plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Violin plot saved to {plot_path}")