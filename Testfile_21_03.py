
from pyexpat import model
from ChosenParameterFitting import fit_parameters, plot_fit, sensitivity_analysis_all, model_accuracy, overall_score
from Dictionary_transforms import compute_means, filter_bounds, restructure_bounds, round_sig
from Models import basic_params, intermediate_params, advanced_params, bounds
import pandas as pd
import numpy as np

# Importing the necessary data from the file
file_path = r"C:\Users\woutg\Documents\Universiteit\Bachelor 3\Bachelorproef\Synthetic_Immune_Data_Memory_Cytokine_Realistic.csv"
df = pd.read_csv(file_path, delimiter=';')

# Compute dictionaries
all_params_bas = compute_means(basic_params, bounds)
all_params_mem = compute_means(intermediate_params, bounds)
all_params_cyt = compute_means(advanced_params, bounds)

bounds_dict_bas = filter_bounds(basic_params, bounds)
bounds_dict_mem = filter_bounds(intermediate_params, bounds)
bounds_dict_cyt = filter_bounds(advanced_params, bounds)

bounds_scipy_bas = restructure_bounds(basic_params, bounds)
bounds_scipy_mem = restructure_bounds(intermediate_params, bounds)
bounds_scipy_cyt = restructure_bounds(advanced_params, bounds)

# compare_conditions(["Normal", "Cancer", "Autoimmune"], df, params_to_compare=[], params_to_skip=["Treg_decay", "Tconv_decay", "Mreg_decay", "K_reg", "tau"], parameters_to_fit=filtered_param_name_list )


conditions = ["Normal","Autoimmune", "Cancer"]
weights = {'R2': 0.5, 'r': 0.5, 'MSE': 0.}


# # Basic Model:
# for condition in conditions:
#     # Fitting parameters
#     selected_params = fit_parameters(
#         condition, 
#         df,
#         all_params_bas,
#         bounds=bounds_scipy_bas,
#         parameters_to_fit=basic_params, 
#         output_format="dict",
#         system = "basic")
    
#     print(f"Fitted parametervalues for basic model in {condition} condition:")
#     for name in basic_params:
#         print(f"{name}: {round_sig(selected_params[name])}")
#     print("\n")
    
#     # Print results of metrics
#     metrics = model_accuracy(condition, df, selected_params, "basic", "all")
#     print("Model Accuracy Metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
#     print("Overall score:",overall_score(metrics, weights, True))
#     print("\n\n")
#     plot_fit(condition, df, param_dict=selected_params, system="basic")
    

# # Memory Model:
# for condition in conditions:
#     # Fitting parameters
#     selected_params = fit_parameters(
#         condition, 
#         df,
#         all_params_mem,
#         bounds=bounds_scipy_mem,
#         parameters_to_fit=intermediate_params, 
#         output_format="dict",
#         system = "memory")

#     print(f"Fitted parametervalues for memory model in {condition} condition:")
#     for name in intermediate_params:
#         print(f"{name}: {round_sig(selected_params[name])}")
#     print("\n")
    
#     # Print results of metrics
#     metrics = model_accuracy(condition, df, selected_params, "memory", "all")
#     print("Model Accuracy Metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
#     print("Overall score:", overall_score(metrics, weights))
#     print("\n\n")
#     plot_fit(condition, df, param_dict=selected_params, system="memory", reg_sum = True)

# # Cytokine Model:
# for condition in conditions:
#     # Fitting parameters
#     selected_params = fit_parameters(
#         condition, 
#         df,
#         all_params_cyt,
#         bounds=bounds_scipy_cyt,
#         parameters_to_fit=advanced_params, 
#         output_format="dict",
#         system = "cytokine")
    
#     print(f"Fitted parametervalues for cytokine model in {condition} condition:")
#     for name in advanced_params:
#         print(f"{name}: {round_sig(selected_params[name])}")
#     print("\n")
    
#     # Print results of metrics
#     metrics = model_accuracy(condition, df, selected_params, "cytokine", "all")
#     print("Model Accuracy Metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
#     print("Overall score:", overall_score(metrics, weights))
#     print("\n\n")
#     plot_fit(condition, df, param_dict=selected_params, system="cytokine")


#########################################
#--- Parameter sensitivity of system ---#
#########################################
variations = np.arange(-0.25, 0.26, 0.025)
variations[np.isclose(variations, 0)] = 0.0
variations = variations.tolist()
sensitivity_analysis_all("Normal", df, "memory", params_to_compare=all_params_mem, metric="r", variations=variations)
sensitivity_analysis_all("Normal", df, "cytokine", params_to_compare=all_params_cyt, variations=variations)

