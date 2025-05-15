
from ChosenParameterFitting import fit_parameters, plot_fit, compare_conditions, sensitivity_analysis_all, sensitivity_analysis_all_data, tornado_plot_sensitivity ,model_accuracy, overall_score
from Dictionary_transforms import compute_means, filter_bounds, restructure_bounds, round_sig
from Models import basic_params, intermediate_params, advanced_params, bounds
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Importing the necessary data from the file
file_path = r"C:\Users\woutg\Documents\Universiteit\Bachelor 3\Bachelorproef\Synthetic_Immune_Data_Memory_Cytokine_Realistic.csv"
df = pd.read_csv(file_path, delimiter=';')


# Extract relevant columns for the Normal condition
time = df["Time (days)"]
tconv_normal = df["Tconv_Normal"]
treg_normal = df["Treg_Normal"]
mreg_normal = df["Mreg_Normal"]
il2_normal = df["IL2_Normal"]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time, tconv_normal, label="Tconv (Normal)", linestyle="-", marker="o")
plt.plot(time, treg_normal, label="Treg (Normal)", linestyle="--", marker="s")
plt.plot(time, mreg_normal, label="Tmreg (Normal)", linestyle=":", marker="d")
plt.plot(time, il2_normal, label="IL-2 (Normal)", linestyle="-.", marker="^")

# Labels and legend
plt.xlabel("Time (Days)")
plt.ylabel("Population (Units/L)")
plt.title("Immune Cell Populations in Normal Condition")
plt.legend()
plt.grid(True)
plt.show()

tconv_cancer = df["Tconv_Cancer"]
treg_cancer = df["Treg_Cancer"]
il2_cancer = df["IL2_Cancer"]
mreg_cancer = df["Mreg_Cancer"]

tconv_cancer = df["Tconv_Autoimmune"]
treg_cancer = df["Treg_Autoimmune"]
il2_cancer = df["IL2_Autoimmune"]
mreg_cancer = df["Mreg_Autoimmune"]


# Set up the 2x2 subplot figure
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Comparison of Immune Population Dynamics: Normal vs. Cancer", fontsize=14)

# Tconv Dynamics
axs[0, 0].plot(time, tconv_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[0, 0].plot(time, tconv_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[0, 0].set_title("Tconv Dynamics")
axs[0, 0].set_xlabel("Time (Days)")
axs[0, 0].set_ylabel("Population Level (Units/L)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Treg Dynamics
axs[0, 1].plot(time, treg_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[0, 1].plot(time, treg_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[0, 1].set_title("Treg Dynamics")
axs[0, 1].set_xlabel("Time (Days)")
axs[0, 1].set_ylabel("Population Level (Units/L)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# IL-2 Dynamics
axs[1, 0].plot(time, il2_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[1, 0].plot(time, il2_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[1, 0].set_title("IL-2 Dynamics")
axs[1, 0].set_xlabel("Time (Days)")
axs[1, 0].set_ylabel("Population Level (Units/L)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Mreg Dynamics
axs[1, 1].plot(time, mreg_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[1, 1].plot(time, mreg_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[1, 1].set_title("Tmreg Dynamics")
axs[1, 1].set_xlabel("Time (Days)")
axs[1, 1].set_ylabel("Population Level (Units/L)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

tconv_normal = df["Tconv_Normal"]
tconv_cancer = df["Tconv_Autoimmune"]

treg_normal = df["Treg_Normal"]
treg_cancer = df["Treg_Autoimmune"]

il2_normal = df["IL2_Normal"]
il2_cancer = df["IL2_Autoimmune"]

mreg_normal = df["Mreg_Normal"]
mreg_cancer = df["Mreg_Autoimmune"]

# Set up the 2x2 subplot figure
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Comparison of Immune Population Dynamics: Normal vs. Autoimmune", fontsize=14)

# Tconv Dynamics
axs[0, 0].plot(time, tconv_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[0, 0].plot(time, tconv_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[0, 0].set_title("Tconv Dynamics")
axs[0, 0].set_xlabel("Time (Days)")
axs[0, 0].set_ylabel("Population Level (Units/L)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Treg Dynamics
axs[0, 1].plot(time, treg_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[0, 1].plot(time, treg_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[0, 1].set_title("Treg Dynamics")
axs[0, 1].set_xlabel("Time (Days)")
axs[0, 1].set_ylabel("Population Level (Units/L)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# IL-2 Dynamics
axs[1, 0].plot(time, il2_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[1, 0].plot(time, il2_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[1, 0].set_title("IL-2 Dynamics")
axs[1, 0].set_xlabel("Time (Days)")
axs[1, 0].set_ylabel("Population Level (Units/L)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Mreg Dynamics
axs[1, 1].plot(time, mreg_normal, label="Normal", color='orange', linestyle='--', marker='o', markersize=3)
axs[1, 1].plot(time, mreg_cancer, label="Cancer", color='red', linestyle='-', marker='x', markersize=3)
axs[1, 1].set_title("Tmreg Dynamics")
axs[1, 1].set_xlabel("Time (Days)")
axs[1, 1].set_ylabel("Population Level (Units/L)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()