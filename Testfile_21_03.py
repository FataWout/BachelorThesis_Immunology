
from ChosenParameterFitting import compare_conditions, fit_parameters, plot_fit, sensitivity_analysis, sensitivity_analysis_all
from ODE_systems import compare_equation_systems, compare_variable_values, heatmap, stochastic_effects
import pandas as pd
import numpy as np

# Importing the necessary data from the file
file_path = r"C:\Users\woutg\Documents\Universiteit\Bachelor 3\Bachelorproef\Synthetic_Immune_Data_Memory_Cytokine_Realistic.csv"
df = pd.read_csv(file_path, delimiter=';')


all_params_bas = {
    "Tconv_suppress_base": 0.018,
    "Tconv_prolif": 0.13,
    "Tconv_decay": 0.05,
    "Treg_recruitment": 0.012,
    "Treg_growth": 0.085,
    "Treg_decay": 0.05,
}


all_params_mem = {
    "Tconv_suppress_base": 0.018,
    "K_reg": 500.0,
    "tau": 5.0,
    "Mreg_conversion_base": 0.05,
    "Tconv_prolif": 0.13,
    "Tconv_decay": 0.05,
    "Treg_recruitment": 0.012,
    "Treg_growth": 0.085,
    "Treg_decay": 0.05,
    "Mreg_growth": 0.05,
    "Mreg_decay": 0.05
}

all_params_cyt = {
    "Tconv_suppress_base": 0.018,
    "Tconv_prolif": 0.13,
    "Tconv_decay": 0.05,
    "Treg_recruitment": 0.012,
    "Treg_growth": 0.085,
    "Treg_decay": 0.05,
    "K_prolif": 10.,
    "K_suppress": 15.,
    "K_recruitment": 12.,
    "K_growth": 8.,
    "IL2_production": 5.2,
    "IL2_consumption": 0.19
}

# Fitting all parameters except decay rates, these are proportional to growth rates
# Thus to look at the proper ratio rather than unnatural values, we keep these constant.
# Comparing the parameters in the different conditions.
filtered_param_name_list_bas = [name for name in list(all_params_bas.keys()) if not name.endswith("decay") and not name =="tau"]
filtered_param_name_list_mem = [name for name in list(all_params_mem.keys()) if not name.endswith("decay") and not name =="tau"]
filtered_param_name_list_cyt = [name for name in list(all_params_cyt.keys()) if not name.endswith("decay") and  not name =="tau"]


# compare_conditions(["Normal", "Cancer", "Autoimmune"], df, params_to_compare=[], params_to_skip=["Treg_decay", "Tconv_decay", "Mreg_decay", "K_reg", "tau"], parameters_to_fit=filtered_param_name_list )


conditions = ["Normal","Autoimmune", "Cancer"]



for condition in conditions:
    
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_bas,
        parameters_to_fit=filtered_param_name_list_bas, 
        output_format="dict",
        system = "basic")

    print(f"Fitted parametervalues for basic model in {condition} condition:")
    for name in filtered_param_name_list_bas:
        print(f"{name}: {selected_params[name]}")
    
    print("\n\n")
    plot_fit(condition, df, param_dict=selected_params, system="basic")


for condition in conditions:
    
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_mem,
        parameters_to_fit=filtered_param_name_list_mem, 
        output_format="dict",
        system = "memory")

    print(f"Fitted parametervalues for memory model in {condition} condition:")
    for name in filtered_param_name_list_mem:
        print(f"{name}: {selected_params[name]}")
    
    print("\n\n")
    plot_fit(condition, df, param_dict=selected_params, system="memory", reg_sum = True)


for condition in conditions:
    
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_cyt,
        parameters_to_fit=filtered_param_name_list_cyt, 
        output_format="dict",
        system = "cytokine")

    print(f"Fitted parametervalues for cytokine model in {condition} condition:")
    for name in filtered_param_name_list_cyt:
        print(f"{name}: {selected_params[name]}")
    
    print("\n\n")
    plot_fit(condition, df, param_dict=selected_params, system="cytokine")


