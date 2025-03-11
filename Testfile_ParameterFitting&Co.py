
from ChosenParameterFitting import compare_conditions, fit_parameters, plot_fit, sensitivity_analysis, sensitivity_analysis_all
from ODE_systems import compare_equation_systems, compare_variable_values, heatmap, stochastic_effects
import pandas as pd
import numpy as np

# Importing the necessary data from the file
file_path = r"C:\Users\woutg\OneDrive\Bureaublad\synthetic_immune_data_final_split.csv"
df = pd.read_csv(file_path, delimiter=';')


# all_params_og = {
#     "Tconv_suppress_base": 0.02,
#     "K_reg": 500.0,
#     "tau": 5.0,
#     "Mreg_conversion_base": 0.005,
#     "Tconv_prolif": 0.12,
#     "Tconv_decay": 0.05,
#     "Treg_recruitment": 0.01,
#     "Treg_growth": 0.08,
#     "Treg_decay": 0.03,
#     "Mreg_growth": 0.01,
#     "Mreg_decay": 0.005
# }
# initial_populations = [100.0, 10.0, 25.0, 2.0]
# timestamps=np.linspace(0, 100, 101)

# # Quick comparison between the two ODE-systems
# print("Now showing a comparison between the two mathematical models.")
# compare_equation_systems(["Memory", "Cytokine"], timestamps)

# # Now seperated to take a look at the mechanics
# print("Now showing the dynamics of the memory model on its own (with refence parameter values).")
# compare_equation_systems(["Memory"], timestamps, plot_Cytokine=False)
# print("Now showing the dynamics of the cytokine model on its own (with refence parameter values).")
# compare_equation_systems(["Cytokine"], timestamps, plot_Mreg=False)

# # Fitting all parameters except decay rates, these are proportional to growth rates
# # Thus to look at the proper ratio rather than unnatural values, we keep these constant.
# # Comparing the parameters in the different conditions.
# param_name_list = list(all_params_og.keys())
# filtered_param_name_list = [name for name in param_name_list if not name.endswith("decay")]

# compare_conditions(["Normal", "Cancer", "Autoimmune"], df, params_to_compare=[], params_to_skip=["Treg_decay", "Tconv_decay", "Mreg_decay", "K_reg", "tau"], parameters_to_fit=filtered_param_name_list )




# # Fit only prolif and other parameters (no decay)
# print("Fit only prolif and other parameters (no decay)")
# selected_params = fit_parameters(
#     "Normal", 
#     df,
#     parameters_to_fit=filtered_param_name_list, 
#     output_format="dict",
#     system = "memory")
# for key, value in selected_params.items():
#     print(f"{key}: {value}")
# print("\n\n")
# selected_params = list(selected_params.values())
# plot_fit("Normal", df, params=selected_params, system="memory")


# # Fit all parameters this time decays aswell.
# all_params = fit_parameters("Normal", df, output_format="dict", parameters_to_fit=None)
# for key, value in all_params.items():
#     print(f"{key}: {value}")
# print("\n\n")
# all_params = list(all_params.values())
# plot_fit("Normal", df, params=all_params)


# # Fit only specific parameters
# print("Fit only specific parameters in normal conditions")
# selected_params = fit_parameters(
#     "Normal", 
#     df,
#     parameters_to_fit=["K_prolif", "K_growth", "K_suppress", "K_recruitment", "Treg_recruitment", "Tconv_suppress_base"], 
#     output_format="dict",
#     system = "cytokine")
# for key, value in selected_params.items():
#     print(f"{key}: {value}")
# print("\n\n")
# selected_params = list(selected_params.values())
# plot_fit("Normal", df, params=selected_params, system="cytokine")


# Fit only specific parameters for cancerous conditions
print("Fit only specific parameters for cancerous conditions")
selected_params = fit_parameters(
    "Cancer", 
    df,
    parameters_to_fit=["K_prolif", "K_growth", "K_suppress", "K_recruitment", "Treg_recruitment", "Tconv_suppress_base"], 
    output_format="dict",
    system = "cytokine")
for key, value in selected_params.items():
    print(f"{key}: {value}")
print("\n\n")
selected_params = list(selected_params.values())
plot_fit("Cancer", df, params=selected_params, system="cytokine")


# # Fit only specific parameters for autoimmune conditions
# print("Fit only specific parameters for autoimmune conditions")
# selected_params = fit_parameters(
#     "Autoimmune", 
#     df,
#     parameters_to_fit=["K_prolif", "K_growth", "K_suppress", "K_recruitment", "Treg_recruitment", "Tconv_suppress_base"], 
#     output_format="dict",
#     system = "cytokine")
# for key, value in selected_params.items():
#     print(f"{key}: {value}")
# print("\n\n")
# selected_params = list(selected_params.values())
# plot_fit("Autoimmune", df, params=selected_params, system="cytokine")



# # Fit some parameters while fixing others
# print("Fit some parameters while fixing others in normal conditions while giving others a different stationary value.")
# fixed_values = {
#     "Tconv_prolif": 0.15,
#     "Tconv_decay": 0.06,
#     "Mreg_decay": 0.008
# }
# mixed_params = fit_parameters(
#     "Normal", 
#     df,
#     parameters_to_fit=["K_prolif", "K_growth", "K_suppress", "K_recruitment", "Treg_recruitment", "Tconv_suppress_base"],
#     fixed_parameters=fixed_values,
#     output_format="dict",
#     system = "cytokine"
# )
# for key, value in mixed_params.items():
#     print(f"{key}: {value}")
# print("\n\n")
# mixed_params = list(mixed_params.values())
# plot_fit("Normal", df, mixed_params)


sensitivity_analysis_all("Autoimmune", df, "cytokine", params_to_compare=["K_suppress", "K_prolif", "K_recruitment", "K_growth"])
variations=[-0.75, -0.25, 0., 0.25, 0.75]