
### This script is where all functions to fit are written

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy
from Models import dT_dt_Advanced_cytokine, dT_dt_Advanced_memory, dT_dt_Basic
param ={}


def fit_parameters(condition, df, initial_guess=None,parameters_to_fit=None, fixed_parameters=None, output_format='dict', system="Memory"):
    """
    Fit model parameters to data.
    
    Args:
        condition: The condition to use for data selection
        df: DataFrame containing the time series data
        parameters_to_fit: List of parameter names to fit (if None, fit all parameters)
        fixed_parameters: Dictionary of fixed parameter values (if None, use default values)
        
    Returns:
        Dictionary of fitted parameter values
    """
    all_params = initial_guess

    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values

    system = system.lower()
    if system == "basic":
        system_num = 0
        y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), np.zeros_like(Tconv_data)])
        
        if initial_guess == None:
            all_params = {
                "Tconv_suppress_base": 0.018,
                "Tconv_prolif": 0.13,
                "Tconv_decay": 0.05,
                "Treg_recruitment": 0.012,
                "Treg_growth": 0.085,
                "Treg_decay": 0.05,
            }

    elif system == "memory":
        system_num = 1
        Mreg_data = df[f"Mreg_{condition}"].values
        y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), Mreg_data])
        
        if initial_guess==None:
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
                
    elif system == "cytokine":
        system_num = 2
        IL2_data = df[f"IL2_{condition}"].values
        y_data = np.vstack([Tconv_data, Treg_data, IL2_data, np.zeros_like(Tconv_data)])
        
        if initial_guess==None:
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
      
    else:
        print("No valid system detected")
        exit 
    
    # If fixed_parameters is provided, update default values
    if fixed_parameters is not None:
        all_params.update(fixed_parameters)
    
    # If parameters_to_fit is not provided, fit all parameters
    if parameters_to_fit is None:
        parameters_to_fit = list(all_params.keys())
    
    # Create a mapping from fit parameter index to parameter name
    fit_param_names = parameters_to_fit
    
    # Initial guess for parameters to be fitted
    initial_guess = [all_params[param] for param in fit_param_names]
    
    def objective(params_to_fit):
        # Create full parameter dictionary with fixed and fitted parameters
        param_dict = all_params.copy()

        # Update with current values of parameters being fitted
        for i, param_name in enumerate(fit_param_names):
            param_dict[param_name] = params_to_fit[i]

        param = all_params
        if system_num == 0:
            sol = solve_ivp(dT_dt_Basic, [time[0], time[-1]], y_data[:, 0], 
                            args=[param_dict], t_eval=time, method='BDF')
        elif system_num == 1:
            sol = solve_ivp(dT_dt_Advanced_memory, [time[0], time[-1]], y_data[:, 0], 
                            args=[param_dict], t_eval=time, method='BDF')
        elif system_num == 2:
            sol = solve_ivp(dT_dt_Advanced_cytokine, [time[0], time[-1]], y_data[:, 0],
                            args=[param_dict], t_eval=time, method='BDF') 

        if sol.y.shape[1] != y_data.shape[1]:
            interp_func = interp1d(sol.t, sol.y, kind='linear', axis=1, fill_value='extrapolate')
            sol_y_interp = interp_func(time)
        else:
            sol_y_interp = sol.y

        squared_difference = np.sum((sol_y_interp - y_data) ** 2)

        ss_total = np.sum((y_data - np.mean(y_data, axis=1, keepdims=True)) ** 2)
        r2_score = 1 - (squared_difference / ss_total)

        return (sol_y_interp - y_data).ravel()
    
    # Set bounds to ensure all parameters are positive
    bounds = (np.zeros(len(initial_guess)), np.inf * np.ones(len(initial_guess)))
    
    # Run optimization
    result = least_squares(objective, initial_guess, bounds=bounds)
    # Process the output based on the requested format
    
    # Create result dictionary with all parameters
    result_dict = all_params.copy()
        
    # Update with fitted parameters
    for i, param_name in enumerate(fit_param_names):
        result_dict[param_name] = result.x[i]
        
    if output_format.lower() == 'dict':
        return result_dict
    elif output_format.lower() == 'list':
        mid_list = list(result_dict.values())
        result_list = [float(value) for value in mid_list]
        return result_list
    else:
        raise ValueError("output_format must be either 'dict' or 'list'")

def plot_fit(condition, df, param_dict=None, system="Memory", reg_sum = False):
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    if param_dict == None:
        param_dict = fit_parameters(condition, df, output_format="dict", system=system)
    system = system.lower()
    
    if system == "basic":
        y0 = [Tconv_data[0], Treg_data[0], 0, 0]

    elif system == "memory":
        Mreg_data = df[f"Mreg_{condition}"].values
        y0 = [Tconv_data[0], Treg_data[0], 0, Mreg_data[0]]
        
    elif system == "cytokine":
        IL2_data = df[f"IL2_{condition}"].values
        y0 = [Tconv_data[0], Treg_data[0], IL2_data[0], 0]
        
    else:
        print("No valid system detected")
        exit 
        
    plt.figure(figsize=(10, 6))
    param = param_dict
    if system == "basic":
        sol = solve_ivp(dT_dt_Basic, [time[0], time[-1]], y0, args=[param_dict], t_eval=time)
    elif system == "memory":
        sol = solve_ivp(dT_dt_Advanced_memory, [time[0], time[-1]], y0, args=[param_dict], t_eval=time)
        plt.plot(time, Mreg_data, 'bo', label='Mreg Data')
        plt.plot(time, sol.y[3], 'b-', label='Mreg Model')
    elif system == "cytokine":
        sol = solve_ivp(dT_dt_Advanced_cytokine, [time[0], time[-1]], y0, args=[param_dict], t_eval=time)
        plt.plot(time, IL2_data, 'bo', label='IL2 Data')
        plt.plot(time, sol.y[2], 'b-', label='IL2 Model')
    
    plt.plot(time, Tconv_data, 'ro', label='Tconv Data')
    plt.plot(time, sol.y[0], 'r-', label='Tconv Model')
    plt.plot(time, Treg_data, 'go', label='Treg Data')
    plt.plot(time, sol.y[1], 'g-', label='Treg Model')

    plt.xlabel('Time (days)')
    plt.ylabel('Cell Population')
    plt.title(f'Fit of $\mathbf{{{system}}}$ ODE Model to $\mathbf{{{condition}}}$ condition Data')
    plt.legend()
    plt.show()

def compare_conditions(conditions, df, params_to_compare=None, params_to_skip=None, parameters_to_fit=None, system="Memory"):
    param_values = {}
    
    for condition in conditions:
        params = fit_parameters(condition, df, parameters_to_fit=parameters_to_fit, system=system, output_format="list")
        param_values[condition] = params
    systemname = system.lower()
    if systemname=="basic":
        param_names = ["Tconv_suppress_base", "Tconv_prolif", "Tconv_decay",
                       "Treg_recruitment", "Treg_growth", "Treg_decay"]
        
    elif systemname=="memory":
        param_names = ["Tconv_suppress_base", "K_reg", "tau", "Mreg_conversion_base", "Tconv_prolif", "Tconv_decay",
                       "Treg_recruitment", "Treg_growth", "Treg_decay", "Mreg_growth", "Mreg_decay"]

    elif systemname=="cytokine":
        param_names = ["Tconv_suppress_base", "Tconv_prolif", "Tconv_decay", "Treg_recruitment", "Treg_growth",
                       "Treg_decay", "K_prolif", "K_suppress", "K_recruitment", "K_growth", "IL2_production", 
                       "IL2_consumption"]
        
    if params_to_compare == None or not params_to_compare:
        params_to_compare = param_names
    filtered_indices = [i for i, name in enumerate(param_names) if name in params_to_compare and name not in params_to_skip]
    filtered_param_names = [param_names[i] for i in filtered_indices]
    df_params = pd.DataFrame({cond: [param_values[cond][i] for i in filtered_indices] for cond in conditions}, index=filtered_param_names)
    
    df_params.plot(kind='bar', figsize=(12, 6))
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.ylabel('Parameter Value')
    plt.title(f'Comparison of Fitted Parameters Across Conditions in $\mathbf{{{system}}}$ ODE-model')
    plt.legend(title="Conditions")
    
    plt.xticks(rotation=45, ha="right")  # Rotate labels and align them to the right
    plt.tight_layout()  # Adjust layout to ensure everything fits well
    plt.show()
    
def sensitivity_analysis(condition, df, parameter_name, system="Memory", variations = [-0.25, -0.10, 0., 0.10, 0.25]):
    """
    Perform sensitivity analysis on a given parameter by varying its value and computing the R^2 score.
    
    Args:
        condition: The condition to use for data selection.
        df: DataFrame containing the time series data.
        parameter_name: The name of the parameter to analyze.
        system: The system model to use ("Memory" or "Cytokine").
        
    Returns:
        DataFrame with varied parameter values and corresponding R^2 scores.
    """
    # Fit parameters to get baseline values
    fitted_params = fit_parameters(condition, df, output_format='dict', system=system)
    if parameter_name not in fitted_params:
        raise ValueError(f"Parameter '{parameter_name}' not found in fitted parameters.")
    
    base_value = fitted_params[parameter_name]

    results = []
    r2_scores = []
    variation_labels = []
    # Extract the data
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    
    if system.lower() == "basic":
        system_func = dT_dt_Basic
        y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), np.zeros_like(Tconv_data)])
    if system.lower() == "memory":
        system_func = dT_dt_Advanced_memory
        Mreg_data = df[f"Mreg_{condition}"].values
        y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), Mreg_data])
    elif system.lower() == "cytokine":
        system_func = dT_dt_Advanced_cytokine
        IL2_data = df[f"IL2_{condition}"].values
        y_data = np.vstack([Tconv_data, Treg_data, IL2_data, np.zeros_like(Tconv_data)])
    else:
        raise ValueError("Invalid system. Choose 'Memory' or 'Cytokine'.")
    
    # Compute R^2 for each variation
    for var in variations:
        modified_params = copy.deepcopy(fitted_params)
        modified_params[parameter_name] = base_value * (1 + var)
        param=modified_params
        # Solve the system
        sol = solve_ivp(system_func, [time[0], time[-1]], y_data[:, 0], args=[modified_params], t_eval=time, method='BDF')
        
        # Interpolate if needed
        if sol.y.shape[1] != y_data.shape[1]:
            interp_func = interp1d(sol.t, sol.y, kind='linear', axis=1, fill_value='extrapolate')
            sol_y_interp = interp_func(time)
        else:
            sol_y_interp = sol.y
        
        # Compute R^2 score
        squared_difference = np.sum((sol_y_interp - y_data) ** 2)
        ss_total = np.sum((y_data - np.mean(y_data, axis=1, keepdims=True)) ** 2)
        r2_score = 1 - (squared_difference / ss_total)
        if r2_score < 0:
            r2_score =0
        variation_label = f"{(1+var)*100:.0f}%"
        print(f"{parameter_name} = {modified_params[parameter_name]} ({(1+var)*100:.0f}%) ; R^2 Score = {r2_score}")
        results.append({"Variation": f"{variation_label}", "R^2 Score": r2_score} )
       
        r2_scores.append(r2_score)
        variation_labels.append(variation_label)
        # Find baseline R^2 (corresponding to variation == 0.0)
    try:
        baseline_index = variations.index(0.0)
        baseline_r2 = r2_scores[baseline_index]
    except ValueError:
        raise ValueError("Variation list must include 0.0 to define baseline R^2.")
    
    # Compute sum of absolute differences from baseline (excluding baseline itself)
    r2_diff_sum = sum(abs(r2 - baseline_r2) for idx, r2 in enumerate(r2_scores) if idx != baseline_index)
    print(f"Sum of absolute R^2 differences from baseline for {parameter_name}: {r2_diff_sum:.4f}")
    
    
    return {
        "Results": pd.DataFrame(results),
        "Sum_R2_Differences": r2_diff_sum
    }

def sensitivity_analysis_all(condition, df, system="Memory", params_to_skip = [], params_to_compare = [], variations = [-0.25, -0.10, 0., 0.10, 0.25]):
    """
    Perform sensitivity analysis on all parameters and plot the R^2 scores.
    
    Args:
        condition: The condition to use for data selection.
        df: DataFrame containing the time series data.
        system: The system model to use ("Memory" or "Cytokine").
        
    Returns:
        None (plots the results).
    """
    fitted_params = fit_parameters(condition, df, output_format='dict', system=system)
    param_names = list(fitted_params.keys())
    if params_to_compare == []:
        params_to_compare = param_names
    results = {}
    for param in param_names:
        if param in params_to_skip:
            continue
        if param in params_to_compare:
            results[param] = {}
            df_results = sensitivity_analysis(condition, df, param, system, variations)
            r2_values = [max(r2, 0) for r2 in df_results["Results"]["R^2 Score"].values]
            results[param]["r2"] = r2_values
            results[param]["r2_tot_diff"] = df_results["Sum_R2_Differences"]
    
    # Plot results
    variations = [str(var * 100) + "%" for var in variations]
    plt.figure(figsize=(12, 6))
    for param, values in results.items():
        plt.plot(variations, values["r2"], marker='o', label=f"{param} ({values['r2_tot_diff']:.4e})")
    
    plt.xlabel("Parameter Variation")
    plt.ylabel("R^2 Score")
    plt.title(f"Sensitivity Analysis for All Parameters ({system} System)")
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    

