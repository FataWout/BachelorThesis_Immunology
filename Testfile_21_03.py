
from ChosenParameterFitting import compare_conditions, fit_parameters, plot_fit, sensitivity_analysis, sensitivity_analysis_all
from Plot_and_Compare_Methods import compare_equation_systems, compare_variable_values, heatmap, stochastic_effects, solve_and_plot_system
from Dictionary_transforms import compute_means, filter_bounds, restructure_bounds, round_sig
from Models import basic_params, intermediate_params, advanced_params
import pandas as pd
import numpy as np

# Importing the necessary data from the file
file_path = r"C:\Users\woutg\Documents\Universiteit\Bachelor 3\Bachelorproef\Synthetic_Immune_Data_Memory_Cytokine_Realistic.csv"
df = pd.read_csv(file_path, delimiter=';')

bounds = {
    "Tconv_suppress_base":  [0.014, 0.28],
    "Tconv_prolif":         [0.01, 0.30],   # 0.13-0.20        
    "Tconv_decay":          [0.05, 0.05],   # 0.05      
    "Treg_recruitment":     [0.012, 0.021], # 0.012-0.021
    "Treg_growth":          [0.06, 0.1],   # 
    "Treg_decay":           [0.03, 0.03],   # 0.03
    "Mreg_growth":          [0.01, 0.04],   # Almost stable with decay
    "Mreg_decay":           [0.02, 0.02],   #
    "Mreg_conversion_base": [0.01, 0.01],   # 0.01
    "K_reg":                [1., 1000.0],   #          
    "tau":                  [5.0, 5.0],     # 5.0
    "K_prolif":             [1., 40.],      # 10-30
    "K_suppress":           [1., 40.],      # 10-30
    "K_recruitment":        [1., 40.],      # 10-30
    "K_growth":             [1., 40.],      # 10-30
    "IL2_production":       [4.2, 6.5],     # 10-30
    "IL2_consumption":      [0.14, 0.23]    # 10-30
    }

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
# Time span
t_span = (0, 1)
t_eval = np.linspace(*t_span, 5000)
y0 = [100, 0, 0, 0]

for condition in conditions:
    
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_bas,
        bounds=bounds_scipy_bas,
        parameters_to_fit=basic_params, 
        output_format="dict",
        system = "basic")

    print(f"Fitted parametervalues for basic model in {condition} condition:")
    for name in basic_params:
        print(f"{name}: {round_sig(selected_params[name])}")
    
    print("\n\n")
    plot_fit(condition, df, param_dict=selected_params, system="basic")
    # solve_and_plot_system(selected_params, y0,t_span, t_eval, "basic")


for condition in conditions:
    
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_mem,
        bounds=bounds_scipy_mem,
        parameters_to_fit=intermediate_params, 
        output_format="dict",
        system = "memory")

    print(f"Fitted parametervalues for memory model in {condition} condition:")
    for name in intermediate_params:
        print(f"{name}: {round_sig(selected_params[name])}")
    
    print("\n\n")
    plot_fit(condition, df, param_dict=selected_params, system="memory", reg_sum = True)
    # solve_and_plot_system(selected_params, y0,t_span, t_eval, "memory")

for condition in conditions:
    
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_cyt,
        bounds=bounds_scipy_cyt,
        parameters_to_fit=advanced_params, 
        output_format="dict",
        system = "cytokine")

    print(f"Fitted parametervalues for cytokine model in {condition} condition:")
    for name in advanced_params:
        print(f"{name}: {round_sig(selected_params[name])}")
    
    print("\n\n")
    plot_fit(condition, df, param_dict=selected_params, system="cytokine")
    # solve_and_plot_system(selected_params, y0,t_span, t_eval, "cytokine")


#########################################
#--- Parameter sensitivity of system ---#
#########################################

sensitivity_analysis_all("Normal", df, "memory", params_to_compare=all_params_mem, variations=[-0.25, -0.2, -0.15, -0.1, -0.05,  0., 0.05, 0.1, 0.15, 0.2, 0.25])
sensitivity_analysis_all("Normal", df, "cytokine", params_to_compare=all_params_cyt, variations=[-0.25, -0.2, -0.15, -0.1, -0.05,  0., 0.05, 0.1, 0.15, 0.2, 0.25])

