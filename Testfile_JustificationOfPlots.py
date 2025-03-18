from PlotOverTime import solve_and_plot_system
from ChosenParameterFitting import fit_parameters, plot_fit, sensitivity_analysis_all
import pandas as pd
import numpy as np

# Importing the necessary data from the file
file_path = r"C:\Users\woutg\OneDrive\Bureaublad\synthetic_immune_data_final_split.csv"
df = pd.read_csv(file_path, delimiter=';')

params_cytokine = ["Tconv_suppress_base", "Tconv_prolif", "Tconv_decay", "Treg_recruitment", "Treg_growth",
                       "Treg_decay", "K_prolif", "K_suppress", "K_recruitment", "K_growth", "IL2_production", 
                       "IL2_consumption"]
params_memory = ["Tconv_suppress_base", "K_reg", "tau", "Mreg_conversion_base", "Tconv_prolif", "Tconv_decay",
                       "Treg_recruitment", "Treg_growth", "Treg_decay", "Mreg_growth", "Mreg_decay"]

#########################################
#--- Parameter sensitivity of system ---#
#########################################

# sensitivity_analysis_all("Normal", df, "memory", params_to_compare=params_memory, variations=[-0.25, -0.2, -0.15, -0.1, -0.05,  0., 0.05, 0.1, 0.15, 0.2, 0.25])
sensitivity_analysis_all("Normal", df, "cytokine", params_to_compare=params_cytokine, variations=[-0.25, -0.2, -0.15, -0.1, -0.05,  0., 0.05, 0.1, 0.15, 0.2, 0.25])












fixed_values = {
    "Mreg_growth": 0.,
    "Mreg_decay": 0.
}

# Fit only prolif and other parameters (no decay)
print("Fit only prolif and other parameters (no decay)")
selected_params = fit_parameters(
    "Normal", 
    df,
    parameters_to_fit=["Tconv_suppress_base",  "tau", "Mreg_conversion_base",
                       "Treg_recruitment", "Treg_growth", "Mreg_growth"], 
    # fixed_parameters = fixed_values,                   
    output_format="dict",
    system = "memory")
for key, value in selected_params.items():
    print(f"{key}: {value}")
print("\n\n")
selected_params_list = list(selected_params.values())
plot_fit("Normal", df, params=selected_params_list, system="memory")


# Initial populations
initial_populations = [100., 10., 5., 2.]  # [Tconv, Treg, IL-2, (dummy 0)]

# Time span
t_span = (0, 500)
t_eval = np.linspace(*t_span, 5000)

# Function call
solve_and_plot_system(selected_params, initial_populations, t_span, t_eval, "memory")

fixed_values = selected_params

# Fit only prolif and other parameters (no decay)
print("Fit only prolif and other parameters (no decay)")
selected_params = fit_parameters(
    "Autoimmune", 
    df,
    parameters_to_fit=["Tconv_suppress_base", "tau", "Mreg_conversion_base",
                       "Treg_recruitment", "Treg_growth", "Mreg_growth"], 
    fixed_parameters = fixed_values,                   
    output_format="dict",
    system = "memory")
for key, value in selected_params.items():
    print(f"{key}: {value}")
print("\n\n")
selected_params_list = list(selected_params.values())

# plot_fit("Autoimmune", df, params=selected_params_list, system="memory")
