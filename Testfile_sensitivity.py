
from ChosenParameterFitting import fit_parameters, plot_fit, compare_conditions, sensitivity_analysis_all, sensitivity_analysis_all_data, tornado_plot_sensitivity ,model_accuracy, overall_score
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

# compare_conditions(["Normal", "Cancer", "Autoimmune"], df, bounds=bounds, params_to_compare=[], params_to_skip=["Treg_decay", "Tconv_decay", "Mreg_decay", "K_reg", "tau"], parameters_to_fit=list(all_params_bas.keys()), system="basic", logplot=False )
# compare_conditions(["Normal", "Cancer", "Autoimmune"], df, bounds=bounds, params_to_compare=[], params_to_skip=["Treg_decay", "Tconv_decay", "Mreg_decay", "K_reg", "tau"], parameters_to_fit=list(all_params_mem.keys()), system="memory", logplot=False )
# compare_conditions(["Normal", "Cancer", "Autoimmune"], df, bounds=bounds, params_to_compare=[], params_to_skip=["Treg_decay", "Tconv_decay", "Mreg_decay", "K_reg", "tau"], parameters_to_fit=list(all_params_cyt.keys()), system="cytokine", logplot=False )


conditions = ["Normal","Autoimmune", "Cancer"]
weights = {'R2': 0.5, 'r': 0.5, 'MSE': 0.}


# Basic Model:
for condition in conditions:
    # Fitting parameters
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_bas,
        bounds_dict=bounds,
        parameters_to_fit=basic_params, 
        output_format="dict",
        system = "basic")
    
    print(f"Fitted parametervalues for basic model in {condition} condition:")
    for name in basic_params:
        print(f"{name}: {round_sig(selected_params[name])}")
    print("\n")
    
    # Print results of metrics
    metrics = model_accuracy(condition, df, selected_params, "basic", "all")
    filtered_metrics = {k: v for k, v in metrics.items() if 'Tconv' in k or 'Treg' in k}
    print("Model Accuracy Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("Overall score:",overall_score(filtered_metrics, weights, True))
    print("\n\n")
    plot_fit(condition, df, param_dict=selected_params, system="basic")
    

# # Memory Model:
# for condition in conditions:
#     # Fitting parameters
#     selected_params = fit_parameters(
#         condition, 
#         df,
#         all_params_mem,
#         bounds_dict=bounds,
#         parameters_to_fit=intermediate_params, 
#         output_format="dict",
#         system = "memory")

#     print(f"Fitted parametervalues for memory model in {condition} condition:")
#     for name in intermediate_params:
#         print(f"{name}: {round_sig(selected_params[name])}")
#     print("\n")
    
#     # Print results of metrics
#     metrics = model_accuracy(condition, df, selected_params, "memory", "all")
#     filtered_metrics = {k: v for k, v in metrics.items() if 'Tconv' in k or 'Treg' in k}
#     print("Model Accuracy Metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
#     print("Overall score:", overall_score(filtered_metrics, weights))
#     print("\n\n")
#     plot_fit(condition, df, param_dict=selected_params, system="memory")

# # Cytokine Model:
# for condition in conditions:
#     # Fitting parameters
#     selected_params = fit_parameters(
#         condition, 
#         df,
#         all_params_cyt,
#         bounds_dict=bounds,
#         parameters_to_fit=advanced_params, 
#         output_format="dict",
#         system = "cytokine")
    
#     print(f"Fitted parametervalues for cytokine model in {condition} condition:")
#     for name in advanced_params:
#         print(f"{name}: {round_sig(selected_params[name])}")
#     print("\n")
    
#     # Print results of metrics
#     metrics = model_accuracy(condition, df, selected_params, "cytokine", "all")
#     filtered_metrics = {k: v for k, v in metrics.items() if 'Tconv' in k or 'Treg' in k}
#     print("Model Accuracy Metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
#     print("Overall score:", overall_score(filtered_metrics, weights))
#     print("\n\n")
#     plot_fit(condition, df, param_dict=selected_params, system="cytokine")


# #########################################
# #--- Parameter sensitivity of system ---#
# #########################################
variations = np.arange(-0.150, 0.151, 0.005)
variations[np.isclose(variations, 0)] = 0.0
variations = variations.tolist()

# # Analysis based on parameters we give it, so the 100% will give rate 1, and other modified parameters will differ. 
# # sensitivity_analysis_all("cytokine", metric="r2", variations=variations)
tornado_plot_sensitivity("Normal", "basic", params_to_compare=all_params_bas, metric="r2")

# # sensitivity_analysis_all("memory", metric="r2", variations=variations)
# tornado_plot_sensitivity("Normal", "memory", params_to_compare=all_params_mem, metric="r2")

# # sensitivity_analysis_all("cytokine", metric="r2", variations=variations)
# tornado_plot_sensitivity("Normal", "cytokine", params_to_compare=all_params_cyt, metric="r2")


# Analysis based on data we give it, so the 100% will possibly give lower values because of boundaries set for parameters. 
# sensitivity_analysis_all_data("Normal", df, "basic", params_to_compare=all_params_bas, metric="r2", variations=variations)
tornado_plot_sensitivity("Normal", "basic", params_to_compare=all_params_bas, metric="r2", df=df, variation=0.001)
tornado_plot_sensitivity("Cancer", "basic", params_to_compare=all_params_bas, metric="r2", df=df, variation=0.001)
tornado_plot_sensitivity("Autoimmune", "basic", params_to_compare=all_params_bas, metric="r2", df=df, variation=0.001)

# sensitivity_analysis_all_data("Normal", df, "memory", params_to_compare=all_params_mem, metric="r2", variations=variations)
tornado_plot_sensitivity("Normal", "memory", params_to_compare=all_params_mem, metric="r2", df=df, variation=0.01)
tornado_plot_sensitivity("Cancer", "memory", params_to_compare=all_params_mem, metric="r2", df=df, variation=0.01)
tornado_plot_sensitivity("Autoimmune", "memory", params_to_compare=all_params_mem, metric="r2", df=df, variation=0.01)

sensitivity_analysis_all_data("Normal", df, "cytokine", params_to_compare=all_params_cyt, metric="r2", variations=variations)
tornado_plot_sensitivity("Normal", "cytokine", params_to_compare=all_params_cyt, metric="r2", df=df, variation=0.01)
tornado_plot_sensitivity("Cancer", "cytokine", params_to_compare=all_params_cyt, metric="r2", df=df, variation=0.01)
tornado_plot_sensitivity("Autoimmune", "cytokine", params_to_compare=all_params_cyt, metric="r2", df=df, variation=0.01)


