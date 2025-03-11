
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

def dT_dt_Advanced_memory(t, populations, param, *args):
    # Fix the unpacking syntax
    Tconv_pop, Treg_pop, _, Mreg_pop = populations
    
    suppress_rate_Tconv = (param["suppress_rate_base_Tconv"] * (1 - Treg_pop / (Treg_pop + param["K_reg"])) 
                                                             * (1 - np.exp(-t/param["tau"])))
    conversion_rate_Mreg = param["conversion_rate_base_Mreg"] * (t / (t + param["tau"]))
    
    dTconv_dt = (param["prolif_rate_Tconv"] * Tconv_pop 
                    - param["decay_rate_Tconv"] * Tconv_pop 
                    - suppress_rate_Tconv * Tconv_pop * (Treg_pop + Mreg_pop) 
                    - param["recruitment_rate_Treg"] * Tconv_pop)
    dTreg_dt = (param["growth_rate_Treg"] * Treg_pop 
                    - param["decay_rate_Treg"] * Treg_pop 
                    + param["recruitment_rate_Treg"] * Tconv_pop 
                    - conversion_rate_Mreg * Treg_pop)
    dMreg_dt = (conversion_rate_Mreg * Treg_pop
                    + param["growth_rate_Mreg"] * Mreg_pop
                    - param["decay_rate_Mreg"] * Mreg_pop)
    
    return np.array([dTconv_dt, dTreg_dt, 0, dMreg_dt])

def dT_dt_Advanced_cytokine(t, populations, param, *args):
    # Fix the unpacking syntax
    Tconv_pop, Treg_pop, IL2, _ = populations
    
    suppress_rate_Tconv = (param["suppress_rate_base_Tconv"] * (1 - Treg_pop / (Treg_pop + param["K_reg"])) 
                                                             * (1 - np.exp(-t/param["tau"])))
    conversion_rate_Mreg = param["conversion_rate_base_Mreg"] * (t / (t + param["tau"]))
    
    dTconv_dt = (param["prolif_rate_Tconv"] * Tconv_pop 
                    - param["decay_rate_Tconv"] * Tconv_pop 
                    - suppress_rate_Tconv * Tconv_pop * (Treg_pop + Mreg_pop) 
                    - param["recruitment_rate_Treg"] * Tconv_pop)
    dTreg_dt = (param["growth_rate_Treg"] * Treg_pop 
                    - param["decay_rate_Treg"] * Treg_pop 
                    + param["recruitment_rate_Treg"] * Tconv_pop 
                    - conversion_rate_Mreg * Treg_pop)
    dIL2_dt = (conversion_rate_Mreg * Treg_pop
                    + param["growth_rate_Mreg"] * Mreg_pop
                    - param["decay_rate_Mreg"] * Mreg_pop)
    
    return np.array([dTconv_dt, dTreg_dt, dIL2_dt, 0])

def fit_parameters(condition, df, parameters_to_fit=None, fixed_parameters=None, output_format='dict', system="Memory"):
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
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    Mreg_data = df[f"Mreg_{condition}"].values
    
    y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), Mreg_data])
    
    # Define all parameters and their default initial values
    all_params = {
        "suppress_rate_base_Tconv": 0.02,
        "K_reg": 500.0,
        "tau": 5.0,
        "conversion_rate_base_Mreg": 0.005,
        "prolif_rate_Tconv": 0.12,
        "decay_rate_Tconv": 0.05,
        "recruitment_rate_Treg": 0.01,
        "growth_rate_Treg": 0.08,
        "decay_rate_Treg": 0.03,
        "growth_rate_Mreg": 0.01,
        "decay_rate_Mreg": 0.005
    }
    
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

        sol = solve_ivp(dT_dt_Advanced_memory, [time[0], time[-1]], y_data[:, 0], args=[param_dict], t_eval=time, method='BDF') #, rtol=1e-3, atol=1e-6)

        if sol.y.shape[1] != y_data.shape[1]:
            interp_func = interp1d(sol.t, sol.y, kind='linear', axis=1, fill_value='extrapolate')
            sol_y_interp = interp_func(time)
        else:
            sol_y_interp = sol.y

        squared_difference = np.sum((sol_y_interp - y_data) ** 2)
        print(f"Sum of squared differences: {squared_difference}")

        ss_total = np.sum((y_data - np.mean(y_data, axis=1, keepdims=True)) ** 2)
        r2_score = 1 - (squared_difference / ss_total)
        print(f"R^2 coefficient of determination: {r2_score}")

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


def plot_fit(condition, df, system="Memory"):
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    Mreg_data = df[f"Mreg_{condition}"].values
    params = fit_parameters(condition, df, output_format="list")
    param_dict = {
        "suppress_rate_base_Tconv": params[0],
        "K_reg": params[1],
        "tau": params[2],
        "conversion_rate_base_Mreg": params[3],
        "prolif_rate_Tconv": params[4],
        "decay_rate_Tconv": params[5],
        "recruitment_rate_Treg": params[6],
        "growth_rate_Treg": params[7],
        "decay_rate_Treg": params[8],
        "growth_rate_Mreg": params[9],
        "decay_rate_Mreg": params[10]
    }
    
    y0 = [Tconv_data[0], Treg_data[0], 0, Mreg_data[0]]
    sol = solve_ivp(dT_dt_Advanced_memory, [time[0], time[-1]], y0, args=[param_dict], t_eval=time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, Tconv_data, 'ro', label='Tconv Data')
    plt.plot(time, sol.y[0], 'r-', label='Tconv Model')
    plt.plot(time, Treg_data, 'go', label='Treg Data')
    plt.plot(time, sol.y[1], 'g-', label='Treg Model')
    plt.plot(time, Mreg_data, 'bo', label='Mreg Data')
    plt.plot(time, sol.y[3], 'b-', label='Mreg Model')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Cell Population')
    plt.title(f'Fit of ODE Model to {condition} Data')
    plt.legend()
    plt.show()


all_params_og = {
    "suppress_rate_base_Tconv": 0.02,
    "K_reg": 500.0,
    "tau": 5.0,
    "conversion_rate_base_Mreg": 0.005,
    "prolif_rate_Tconv": 0.12,
    "decay_rate_Tconv": 0.05,
    "recruitment_rate_Treg": 0.01,
    "growth_rate_Treg": 0.08,
    "decay_rate_Treg": 0.03,
    "growth_rate_Mreg": 0.01,
    "decay_rate_Mreg": 0.005
}
file_path = r"C:\Users\woutg\OneDrive\Bureaublad\synthetic_immune_data_final_split.csv"
df = pd.read_csv(file_path, delimiter=';')

plot_fit("Normal", df, list(all_params_og.values()))

# Fit all parameters (same as before)
# all_params = fit_parameters("Autoimmune", df, output_format="list")
plot_fit("Autoimmune", df)

# Fit only specific parameters
selected_params = fit_parameters(
    "Normal", 
    df,
    parameters_to_fit=["suppress_rate_base_Tconv"], 
    output_format="list")
plot_fit("Normal", df, selected_params)

# Fit some parameters while fixing others
fixed_values = {
    "prolif_rate_Tconv": 0.15,
    "decay_rate_Tconv": 0.06,
    "decay_rate_Mreg": 0.008
}
mixed_params = fit_parameters(
    "Normal", 
    df,
    parameters_to_fit=["suppress_rate_base_Tconv", "K_reg", "tau", "growth_rate_Treg", "conversion_rate_base_Mreg"],
    fixed_parameters=fixed_values,
    output_format="list"
)
plot_fit("Normal", df, mixed_params)