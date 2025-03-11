import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

def dT_dt_Advanced_memory(t, populations, param, *args):
    Tconv_pop, Treg_pop, _, Mreg_pop = populations
    
    suppress_rate_Tconv = param["suppress_rate_base_Tconv"] * (1 - Treg_pop / (Treg_pop + param["K_reg"])) * (1 - np.exp(-t/param["tau"]))
    conversion_rate_Mreg = param["conversion_rate_base_Mreg"] * (t / (t +  param["tau"]))
    
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

def fit_parameters(condition, df):
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    Mreg_data = df[f"Mreg_{condition}"].values
    
    y_data = np.vstack([Tconv_data, Treg_data, np.zeros_like(Tconv_data), Mreg_data])
    
    def objective(params):
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
        
        sol = solve_ivp(dT_dt_Advanced_memory, [time[0], time[-1]], y_data[:, 0], args=[param_dict], t_eval=time)
        
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
    
    initial_guess = [0.02, 500., 5., 0.005, 0.12, 0.05, 0.01, 0.08, 0.03, 0.01, 0.005]
    bounds = (np.zeros(11), np.inf)  # Ensure all parameters are positive
    result = least_squares(objective, initial_guess, bounds=bounds)
    
    return result.x

def plot_fit(condition, df, params):
    time = df["Time (days)"].values
    Tconv_data = df[f"Tconv_{condition}"].values
    Treg_data = df[f"Treg_{condition}"].values
    Mreg_data = df[f"Mreg_{condition}"].values
    
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


def compare_conditions(conditions, df, skip_params=[]):
    param_values = {}
    
    for condition in conditions:
        params = fit_parameters(condition, df)
        param_values[condition] = params
    
    param_names = ["suppress_rate_base_Tconv", "K_reg", "tau", "conversion_rate_base_Mreg", "prolif_rate_Tconv", "decay_rate_Tconv", "recruitment_rate_Treg", "growth_rate_Treg", "decay_rate_Treg", "growth_rate_Mreg", "decay_rate_Mreg"]
    
    filtered_indices = [i for i, name in enumerate(param_names) if name not in skip_params]
    filtered_param_names = [param_names[i] for i in filtered_indices]
    df_params = pd.DataFrame({cond: [param_values[cond][i] for i in filtered_indices] for cond in conditions}, index=filtered_param_names)
    
    df_params.plot(kind='bar', figsize=(12, 6))
    plt.ylabel('Parameter Value')
    plt.title('Comparison of Fitted Parameters Across Conditions')
    plt.legend(title="Conditions")
    
    plt.xticks(rotation=45, ha="right")  # Rotate labels and align them to the right
    plt.tight_layout()  # Adjust layout to ensure everything fits well
    plt.show()


initial_populations = [100.0, 10.0, 25.0, 2.0]
timestamps=np.linspace(0, 100, 101)
# Load the data
file_path = r"C:\Users\woutg\OneDrive\Bureaublad\synthetic_immune_data_final_split.csv"
df = pd.read_csv(file_path, delimiter=';')
condition = "Cancer"
params_found = fit_parameters(condition, df)
print(params_found)
plot_fit(condition, df, params_found)
compare_conditions(["Normal", "Cancer", "Autoimmune"], df,["tau", "K_reg"])
