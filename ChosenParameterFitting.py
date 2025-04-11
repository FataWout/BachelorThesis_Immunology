
### This script is where all functions to fit are written

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy
from scipy.stats import pearsonr
from Dictionary_transforms import restructure_bounds, compute_means
from Models import dT_dt_Advanced_cytokine, dT_dt_Advanced_memory, dT_dt_Basic, basic_params, intermediate_params, advanced_params, bounds
param ={}


def fit_parameters(condition, df, initial_guess=None, parameters_to_fit=None, bounds_dict=None, fixed_parameters=None, output_format='dict', system="memory"):
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
    system = system.lower()
    condition = condition.lower().capitalize()
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    if bounds_dict==None:
        bounds_dict = bounds

    if system == "basic":
        system_num = 0
        y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), np.zeros_like(Tconv_data)])
        
        if initial_guess == None:
            all_params = compute_means(basic_params, bounds)
        bounds_scipy = restructure_bounds(basic_params, bounds)
    elif system == "memory":
        system_num = 1
        Mreg_data = df[f"Mreg_{condition}"].values
        y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), Mreg_data])
        
        if initial_guess==None:
            all_params = compute_means(intermediate_params, bounds)
        bounds_scipy = restructure_bounds(intermediate_params, bounds)
                
    elif system == "cytokine":
        system_num = 2
        IL2_data = df[f"IL2_{condition}"].values
        y_data = np.vstack([Tconv_data, Treg_data, IL2_data, np.zeros_like(Tconv_data)])
        
        if initial_guess==None:
            all_params = compute_means(advanced_params, bounds)
        bounds_scipy = restructure_bounds(advanced_params, bounds)
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
        return (sol_y_interp - y_data).ravel()
    

    # Run optimization
    result = least_squares(objective, initial_guess, bounds=bounds_scipy)
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
    system = system.lower()   

    if param_dict == None:
        param_dict = fit_parameters(condition, df, output_format="dict", system=system)
        
    plt.figure(figsize=(10, 6))
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    if system == "basic":
        y0 = [Tconv_data[0], Treg_data[0], 0, 0]
        sol = solve_ivp(dT_dt_Basic, [time[0], time[-1]], y0, args=[param_dict], t_eval=time)
        reg_sum = False
    elif system == "memory":
        Mreg_data = df[f"Mreg_{condition}"].values
        y0 = [Tconv_data[0], Treg_data[0], 0, Mreg_data[0]]
        sol = solve_ivp(dT_dt_Advanced_memory, [time[0], time[-1]], y0, args=[param_dict], t_eval=time)
        if reg_sum == True:
            plt.plot(time, Mreg_data + Treg_data, 'bo', label='Mreg Data')
            plt.plot(time, sol.y[3] + sol.y[1], 'b-', label='Mreg Model')
        else:
            plt.plot(time, Mreg_data, 'bo', label='Mreg Data')
            plt.plot(time, sol.y[3], 'b-', label='Mreg Model')
    elif system == "cytokine":
        IL2_data = df[f"IL2_{condition}"].values
        y0 = [Tconv_data[0], Treg_data[0], IL2_data[0], 0]
        sol = solve_ivp(dT_dt_Advanced_cytokine, [time[0], time[-1]], y0, args=[param_dict], t_eval=time)
        plt.plot(time, IL2_data, 'bo', label='IL2 Data')
        plt.plot(time, sol.y[2], 'b-', label='IL2 Model')
        reg_sum = False
    else:
        raise ValueError("No valid system given.")        
    
    plt.plot(time, Tconv_data, 'ro', label='Tconv Data')
    plt.plot(time, sol.y[0], 'r-', label='Tconv Model')
    if reg_sum == False:
        plt.plot(time, Treg_data, 'go', label='Treg Data')
        plt.plot(time, sol.y[1], 'g-', label='Treg Model')

    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (Units/L)')
    plt.title(f'Fit of $\mathbf{{{system.capitalize()}}}$ ODE Model to $\mathbf{{{condition.capitalize()}}}$ Condition Data')
    plt.legend()
    plt.show()

def compare_conditions(conditions, df, bounds=bounds, params_to_compare=None, params_to_skip=None, parameters_to_fit=None, system="Memory", logplot=True):
    param_values = {}
    
    for condition in conditions:
        params = fit_parameters(condition, df, bounds_dict=bounds, parameters_to_fit=parameters_to_fit, system=system, output_format="list")
        param_values[condition] = params
    systemname = system.lower()
    if systemname=="basic":
        param_names = basic_params
        
    elif systemname=="memory":
        param_names = intermediate_params

    elif systemname=="cytokine":
        param_names = advanced_params
        
    if params_to_compare == None or not params_to_compare:
        params_to_compare = param_names
    filtered_indices = [i for i, name in enumerate(param_names) if name in params_to_compare and name not in params_to_skip]
    filtered_param_names = [param_names[i] for i in filtered_indices]
    df_params = pd.DataFrame({cond: [param_values[cond][i] for i in filtered_indices] for cond in conditions}, index=filtered_param_names)
    
    ax = df_params.plot(kind='bar', figsize=(12, 6))
    # Draw a limit line for each parameter
    for i, param in enumerate(filtered_param_names):
        if param in bounds:
            limit = bounds[param]  
            ax.hlines(y=limit[0], xmin=i - 0.4, xmax=i + 0.4, colors='red', linestyles='dashed') 
            ax.hlines(y=limit[1], xmin=i - 0.4, xmax=i + 0.4, colors='red', linestyles='dashed') 
    if logplot==True:
        plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.ylabel('Parameter Value')
    plt.title(f'Comparison of Fitted Parameters Across Conditions in $\mathbf{{{system}}}$ ODE-model')
    plt.legend(title="Conditions")
    
    plt.xticks(rotation=45, ha="right")  # Rotate labels and align them to the right
    plt.tight_layout()  # Adjust layout to ensure everything fits well
    plt.show()
    
def sensitivity_analysis(condition, df, parameter_name, params=None, system="Memory", metric='r', variations = [-0.25, -0.10, 0., 0.10, 0.25]):
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
    system = system.lower()
    if params == None:
        if system == "basic":
            all_params = compute_means(basic_params)
            bounds_scipy = restructure_bounds(basic_params)
        elif system == "memory":
            all_params = compute_means(intermediate_params)
            bounds_scipy = restructure_bounds(intermediate_params)
        elif system == "cytokine":
            all_params = compute_means(advanced_params)
            bounds_scipy = restructure_bounds(advanced_params)
        params = fit_parameters(condition, df, all_params, bounds=bounds_scipy ,output_format='dict', system=system)
        
    if parameter_name not in params:
        raise ValueError(f"Parameter '{parameter_name}' not found in fitted parameters.")
    
    base_value = params[parameter_name]

    results = []
    metric_scores = []
    variation_labels = []
    
    # Compute R^2 for each variation
    for var in variations:
        modified_params = copy.deepcopy(params)
        modified_params[parameter_name] = base_value * (1 + var)
        scores = model_accuracy(condition, df, modified_params, system=system, metric=metric, intergration_method="BDF")
  
        metric_score = np.mean(list(scores.values()))
        

        if metric_score < 0:
            metric_score =0
        variation_label = f"{(1+var)*100:.0f}%"
        print(f"{parameter_name} = {modified_params[parameter_name]} ({(1+var)*100:.0f}%) ; {metric.capitalize()} Score = {metric_score}")
        results.append({"Variation": f"{variation_label}", "Score": metric_score} )
       
        metric_scores.append(metric_score)
        variation_labels.append(variation_label)
        # Find baseline R^2 (corresponding to variation == 0.0)
    try:
        baseline_index = variations.index(0.0)
        baseline_score= metric_scores[baseline_index]
    except ValueError:
        raise ValueError("Variation list must include 0.0 to define baseline R^2.")
    
    # Compute sum of absolute differences from baseline (excluding baseline itself)
    diff_sum = sum(abs(score - baseline_score) for idx, score in enumerate(metric_scores) if idx != baseline_index)
    print(f"Sum of absolute {metric.capitalize()} differences from baseline for {parameter_name}: {diff_sum:.4f}")
    
    
    return {
        "Results": pd.DataFrame(results),
        "Sum_Differences": diff_sum
    }

def sensitivity_analysis_all(system="memory", y0=[100,1,25,1], t_span=[0, 100], t_eval=None, params=None, params_to_skip = [], params_to_compare = [], metric='r', variations = [-0.25, -0.10, 0., 0.10, 0.25], show_plot=True):
    """
    Perform sensitivity analysis on all parameters and plot the R^2 scores.
    
    Args:
        condition: The condition to use for data selection.
        df: DataFrame containing the time series data.
        system: The system model to use ("Memory" or "Cytokine").
        
    Returns:
        None (plots the results).
    """
    system = system.lower()
    metric = metric.lower()
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)  # Default time resolution

    if params == None:
        if system == "basic":
            params = compute_means(basic_params)
        elif system == "memory":
            params = compute_means(intermediate_params)
        elif system == "cytokine":
            params = compute_means(advanced_params)
        
    param_names = list(params.keys())
    if params_to_compare == []:
        params_to_compare = param_names
    
    # Define ODE system with parameters
    if system == "basic":
        def ode_system(t, y):
            return dT_dt_Basic(t, y, params)    
        def ode_system_mod(t, y):
            return dT_dt_Basic(t, y, modified_params) 
    elif system == "memory":
        def ode_system(t, y):
            return dT_dt_Advanced_memory(t, y, params)
        def ode_system_mod(t, y):
            return dT_dt_Advanced_memory(t, y, modified_params) 
    elif system == "cytokine":
        def ode_system(t, y):
            return dT_dt_Advanced_cytokine(t, y, params)     
        def ode_system_mod(t, y):
            return dT_dt_Advanced_cytokine(t, y, modified_params) 

    # Solve ODE system
    reference_sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45')
   
    results_vars = {}
    for param in param_names:
        if param in params_to_compare and param not in params_to_skip:
            results_vars[param] = []
            # Compute R^2 for each variation
            for var in variations:
                if float(var) == 0.:
                    results_vars[param].append(1.)
                    continue
                
                modified_params = copy.deepcopy(params)
                modified_params[param] = params[param] * (1 + var)
                
                compare_sol = solve_ivp(ode_system_mod, t_span, y0, t_eval=t_eval, method='RK45')
                
                # Extract model predictions
                Tconv_data, Tconv_model = reference_sol.y[0], compare_sol.y[0]
                Treg_data, Treg_model = reference_sol.y[1], compare_sol.y[1]
                IL2_data, IL2_model = reference_sol.y[2], compare_sol.y[2]
                Mreg_data, Mreg_model = reference_sol.y[3], compare_sol.y[3]
                # Compute chosen metric
                # Compute metrics
                results = {}

                if metric.lower() in ["mse"]:
                    results["MSE_Tconv"] = np.mean((Tconv_model - Tconv_data) ** 2)
                    results["MSE_Treg"] = np.mean((Treg_model - Treg_data) ** 2)
                    if system == "cytokine":
                        results["MSE_IL2"] = np.mean((IL2_model - IL2_data) ** 2)
                    if system == "memory":
                        results["MSE_Mreg"] = np.mean((Mreg_model - Mreg_data) ** 2)
    
                if metric.lower() in ["r"]:
                    results["r_Tconv"], _ = pearsonr(Tconv_model, Tconv_data)
                    results["r_Treg"], _ = pearsonr(Treg_model, Treg_data)
                    if system == "cytokine":
                        results["r_IL2"], _ = pearsonr(IL2_model, IL2_data)
                    if system == "memory":
                        results["r_Mreg"], _ = pearsonr(Mreg_model, Mreg_data)

                if metric.lower() in ["r2"]:
                    ss_tot_Tconv = np.sum((Tconv_data - np.mean(Tconv_data)) ** 2)
                    ss_res_Tconv = np.sum((Tconv_data - Tconv_model) ** 2)
                    results["R2_Tconv"] = max(0., 1 - (ss_res_Tconv / ss_tot_Tconv))
        
                    ss_tot_Treg = np.sum((Treg_data - np.mean(Treg_data)) ** 2)
                    ss_res_Treg = np.sum((Treg_data - Treg_model) ** 2)
                    results["R2_Treg"] = max(0., 1 - (ss_res_Treg / ss_tot_Treg))
        
                    if system == "cytokine":
                        ss_tot_IL2 = np.sum((IL2_data - np.mean(IL2_data)) ** 2)
                        ss_res_IL2 = np.sum((IL2_data - IL2_model) ** 2)
                        results["R2_IL2"] = max(0., 1 - (ss_res_IL2 / ss_tot_IL2))
                    if system == "memory":
                        print(np.mean(Mreg_data))
                        ss_tot_Mreg = np.sum((Mreg_data - np.mean(Mreg_data)) ** 2)
                        ss_res_Mreg = np.sum((Mreg_data - Mreg_model) ** 2)
                        results["R2_Mreg"] = max(0., 1 - (ss_res_Mreg / ss_tot_Mreg))
    
                if metric.lower() not in ["mse", "r", "r2"]:
                    raise ValueError("Invalid metric chosen. Choose 'mse', 'r', or 'r2'.")
                
                metric_score = np.mean(list(results.values()))
                results_vars[param].append(metric_score)
                print(f"{param} = {modified_params[param]:.4e} ({(1+var)*100:.0f}%) ; {metric.capitalize()} Score = {metric_score}")
                
    # Plot results
    variations_labels = [f"{var * 100:.1f}%" for var in variations]
    if show_plot == True:
        plt.figure(figsize=(12, 6))
        for param, values in results_vars.items():
            plt.plot(variations_labels, values, marker='o', markersize=2, label=f"{param} ({params[param]:.2e})")
    
        plt.xlabel("Parameter Variation")
        plt.ylabel(f"{metric.capitalize()} Score")
        plt.title(f"Sensitivity Analysis for All Parameters ({system.capitalize()} System)")
        plt.legend(loc='best', bbox_to_anchor=(1, 1))
        xtick_positions = np.linspace(0, len(variations) - 1, 6, dtype=int)
        xtick_labels = [variations_labels[i] for i in xtick_positions]
        plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=45)
        plt.grid(True)
        plt.show()
    return results_vars

def sensitivity_analysis_all_data(condition, df, system="memory", params=None, bounds=None, params_to_skip = [], params_to_compare = [], metric='r', variations = [-0.25, -0.10, 0., 0.10, 0.25], show_plot=True):
    """
    Perform sensitivity analysis on all parameters and plot the R^2 scores.
    
    Args:
        condition: The condition to use for data selection.
        df: DataFrame containing the time series data.
        system: The system model to use ("Memory" or "Cytokine").
        
    Returns:
        None (plots the results).
    """
    system = system.lower()
    metric = metric.lower()
    condition = condition.lower().capitalize()
    
    if params == None:
        if system == "basic":
            all_params = compute_means(basic_params)
        elif system == "memory":
            all_params = compute_means(intermediate_params)
        elif system == "cytokine":
            all_params = compute_means(advanced_params)
        params = fit_parameters(condition, df, all_params, bounds_dict=bounds, output_format='dict', system=system)
        
    param_names = list(params.keys())
    if params_to_compare == []:
        params_to_compare = param_names
        
    results = {}
    for param in param_names:
        if param in params_to_compare and param not in params_to_skip:
            results[param] = {}
            df_results = sensitivity_analysis(condition, df, param, params=params, system=system, metric=metric, variations=variations)
            score_values = [max(score, 0) for score in df_results["Results"]["Score"].values]
            results[param] = score_values

    
    # Plot results
    variations_labels = [f"{var * 100:.1f}%" for var in variations]
    if show_plot==True:
        plt.figure(figsize=(12, 6))
        for param, values in results.items():
            plt.plot(variations_labels, values, marker='o', markersize=2, label=f"{param} ({params[param]:.2e})")
    
        plt.xlabel("Parameter Variation")
        plt.ylabel(f"{metric.capitalize()} Score")
        plt.title(f"Sensitivity Analysis for All Parameters ({system.capitalize()} System)")
        plt.legend(loc='best', bbox_to_anchor=(1, 1))
        xtick_positions = np.linspace(0, len(variations) - 1, 6, dtype=int)
        xtick_labels = [variations_labels[i] for i in xtick_positions]
        plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=45)
        plt.grid(True)
        plt.show()
    return results
    
def tornado_plot_sensitivity(condition, system="memory", params=None, params_to_skip=[], params_to_compare=[], metric='r', df=None, variation=0.075):
    """
    Generate a symmetric tornado plot showing effect of +/-20% parameter variation on the metric.
    Bars extend left (for -20%) and right (for +20%) from a zero baseline.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    system = system.lower()
    metric = metric.lower()
    condition = condition.lower().capitalize()

    if params is None:
        if system == "basic":
            all_params = compute_means(basic_params)
        elif system == "memory":
            all_params = compute_means(intermediate_params)
        elif system == "cytokine":
            all_params = compute_means(advanced_params)
        if df is not None:
            all_params = fit_parameters(condition, df, all_params, output_format='dict', system=system)
    else:
        all_params = params

    param_names = list(all_params.keys())
    if not params_to_compare:
        params_to_compare = param_names

    variations = [-variation, 0.0, variation]

    labels = []
    results = []
    if df is not None:
        sens_results = sensitivity_analysis_all_data(condition, df, system, all_params, params_to_compare=param_names, metric=metric, variations=variations, show_plot=False)
        print(sens_results)
    else:
        sens_results = sensitivity_analysis_all(system, params_to_compare=param_names, metric=metric, variations=variations, show_plot=False)

    for param in param_names:
        if param in params_to_compare and param not in params_to_skip:
            scores = [max(score, 0) for score in sens_results[param]]
            base_score = scores[1]
            neg_diff = scores[0] - base_score
            pos_diff = scores[2] - base_score

            # Store diffs as signed values to plot: left = -20%, right = +20%
            results.append((neg_diff, pos_diff))    
            labels.append(f"{param}")

    # Sort by total absolute impact
    total_impacts = [abs(neg) + abs(pos) for neg, pos in results]
    sorted_indices = np.argsort(total_impacts)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(10, len(sorted_labels) * 0.4))
    if metric != "mse":
        plt.gca().invert_xaxis()
    for i, (neg_diff, pos_diff) in enumerate(sorted_results):
        if abs(pos_diff) >= abs(neg_diff):
            plt.barh(i, width=pos_diff, left=0, color='skyblue', edgecolor='black', zorder=1, label=f'+{variation*100}%' if i == 0 else "")
            plt.barh(i, width=neg_diff, left=0, color='salmon', edgecolor='black', zorder=2, label=f'-{variation*100}%' if i == 0 else "")
        else:
            plt.barh(i, width=neg_diff, left=0, color='salmon', edgecolor='black', zorder=1, label=f'-{variation*100}%' if i == 0 else "")
            plt.barh(i, width=pos_diff, left=0, color='skyblue', edgecolor='black', zorder=2, label=f'+{variation*100}%' if i == 0 else "")

    
    plt.yticks(range(len(sorted_labels)), sorted_labels)
    plt.xlabel(f"Δ{metric.upper()} Score")
    if df is not None:
       plt.title(
                rf"$\mathbf{{\Delta {metric.upper()} \ Sensitivity\ Ranking}}$: {system.capitalize()} System in {condition.capitalize()} Conditions")

    else:
        plt.title(f"Δ{metric.upper()} Sensitivity Ranking ({system.capitalize()} System)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def model_accuracy(condition, df, param_dict=None, system="Memory", metric ="r", intergration_method="RK45"):
    system = system.lower()
    
    time = df["Time (days)"].values 
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    
    if param_dict == None:
        param_dict = fit_parameters(condition, df, output_format="dict", system=system)
    
    
    if system == "basic":
        y0 = [Tconv_data[0], Treg_data[0], 0, 0]
        sol = solve_ivp(dT_dt_Basic, [time[0], time[-1]], y0, args=[param_dict], t_eval=time, method=intergration_method )

    elif system == "memory":
        Mreg_data = df[f"Mreg_{condition}"].values
        y0 = [Tconv_data[0], Treg_data[0], 0, Mreg_data[0]]
        sol = solve_ivp(dT_dt_Advanced_memory, [time[0], time[-1]], y0, args=[param_dict], t_eval=time, method=intergration_method)
        
    elif system == "cytokine":
        IL2_data = df[f"IL2_{condition}"].values
        y0 = [Tconv_data[0], Treg_data[0], IL2_data[0], 0]
        sol = solve_ivp(dT_dt_Advanced_cytokine, [time[0], time[-1]], y0, args=[param_dict], t_eval=time, method=intergration_method)
        
    else:
        raise ValueError("No valid system given.")
    
    # Extract model predictions
    Tconv_model = sol.y[0]
    Treg_model = sol.y[1]
    IL2_model = sol.y[2]
    Mreg_model = sol.y[3]
    # Compute chosen metric
    # Compute metrics
    results = {}

    if metric.lower() in ["mse", "all"]:
        results["MSE_Tconv"] = np.mean((Tconv_model - Tconv_data) ** 2)
        results["MSE_Treg"] = np.mean((Treg_model - Treg_data) ** 2)
        if system == "cytokine":
            results["MSE_IL2"] = np.mean((IL2_model - IL2_data) ** 2)
        if system == "memory":
            results["MSE_Mreg"] = np.mean((Mreg_model - Mreg_data) ** 2)
    
    if metric.lower() in ["r", "all"]:
        results["r_Tconv"], _ = pearsonr(Tconv_model, Tconv_data)
        results["r_Treg"], _ = pearsonr(Treg_model, Treg_data)
        if system == "cytokine":
            results["r_IL2"], _ = pearsonr(IL2_model, IL2_data)
        if system == "memory":
            results["r_Mreg"], _ = pearsonr(Mreg_model, Mreg_data)

    if metric.lower() in ["r2", "all"]:
        ss_tot_Tconv = np.sum((Tconv_data - np.mean(Tconv_data)) ** 2)
        ss_res_Tconv = np.sum((Tconv_data - Tconv_model) ** 2)
        results["R2_Tconv"] = 1 - (ss_res_Tconv / ss_tot_Tconv)
        
        ss_tot_Treg = np.sum((Treg_data - np.mean(Treg_data)) ** 2)
        ss_res_Treg = np.sum((Treg_data - Treg_model) ** 2)
        results["R2_Treg"] = 1 - (ss_res_Treg / ss_tot_Treg)
        
        if system == "cytokine":
            ss_tot_IL2 = np.sum((IL2_data - np.mean(IL2_data)) ** 2)
            ss_res_IL2 = np.sum((IL2_data - IL2_model) ** 2)
            results["R2_IL2"] = 1 - (ss_res_IL2 / ss_tot_IL2)
        if system == "memory":
            ss_tot_Mreg = np.sum((Mreg_data - np.mean(Mreg_data)) ** 2)
            ss_res_Mreg = np.sum((Mreg_data - Mreg_model) ** 2)
            results["R2_Mreg"] = 1 - (ss_res_Mreg / ss_tot_Mreg)
    
    if metric.lower() not in ["mse", "r", "r2", "all"]:
        raise ValueError("Invalid metric chosen. Choose 'mse', 'r', or 'r2'.")
    return results

def overall_score(metrics: dict, weights=None, only_main_pops = False):
    """
    Calculate the overall score of a model based on a variable number of MSE, r, and R2 values.
    metrics: Dictionary containing the model's metrics (MSE, r, R2 values).
    weights: Optional dictionary to apply custom weights for R2, r, and MSE.
    """
    if weights is None:
        weights = {'R2': 0.4, 'r': 0.4, 'MSE': 0.2}  # Default weights

    # Separate the metrics into MSE, r, and R2 values
    MSE_values = [value for key, value in metrics.items() if key.startswith('MSE')]
    r_values = [value for key, value in metrics.items() if key.startswith('r')]
    R2_values = [value for key, value in metrics.items() if key.startswith('R2')]

    if only_main_pops == True:
        MSE_values = [metrics["MSE_Tconv"], metrics["MSE_Treg"]]
        r_values = [metrics["r_Tconv"], metrics["r_Treg"]]
        R2_values = [metrics["R2_Tconv"], metrics["R2_Treg"]]
    # Calculate the means of each metric
    mean_MSE = np.mean(MSE_values) if MSE_values else 0
    mean_r = np.mean(r_values) if r_values else 0
    mean_R2 = np.mean(R2_values) if R2_values else 0

    # Normalizing MSE
    normalized_MSE = 1 / (1 + mean_MSE) if mean_MSE else 0

    # Calculate the overall score
    overall_score = (weights['R2'] * mean_R2 +
                     weights['r'] * mean_r -
                     weights['MSE'] * normalized_MSE)

    return overall_score
