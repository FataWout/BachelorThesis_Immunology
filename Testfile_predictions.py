
from ChosenParameterFitting import fit_parameters
from Dictionary_transforms import compute_means, filter_bounds, restructure_bounds, round_sig
from Models import basic_params, intermediate_params, advanced_params, bounds
from Plot_and_Compare_Methods import solve_and_plot_system
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

# Define pre-peak initial conditions (modify these if you have better biological estimates)
# [Tconv, Treg, IL2, Tmreg]
y0_pre_peak = [1, 0, 100.0, 0.0]  
conditions = ["Normal","Autoimmune", "Cancer"]


# Time span: simulate from time 0 to e.g. 200 (adjust if needed)
t_span = (0, 200)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Loop over conditions, fit parameters and simulate
for condition in conditions:
    print(f"Running simulation for: {condition}")

    # Fit parameters for current condition
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_bas,
        bounds_dict=bounds,
        parameters_to_fit=basic_params, 
        output_format="dict",
        system="basic"
    )

    # Solve and plot dynamics using the fitted parameters and initial conditions
    sol = solve_and_plot_system(
        selected_params, 
        y0=y0_pre_peak, 
        t_span=t_span, 
        t_eval=t_eval, 
        system="basic"
    )
    
for condition in conditions:
    print(f"Running simulation for: {condition}")

    # Fit parameters for current condition
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_mem,
        bounds_dict=bounds,
        parameters_to_fit=intermediate_params, 
        output_format="dict",
        system="memory"
    )

    # Solve and plot dynamics using the fitted parameters and initial conditions
    sol = solve_and_plot_system(
        selected_params, 
        y0=y0_pre_peak, 
        t_span=t_span, 
        t_eval=t_eval, 
        system="memory"
    )

for condition in conditions:
    print(f"Running simulation for: {condition}")

    # Fit parameters for current condition
    selected_params = fit_parameters(
        condition, 
        df,
        all_params_cyt,
        bounds_dict=bounds,
        parameters_to_fit=advanced_params, 
        output_format="dict",
        system="cytokine"
    )

    # Solve and plot dynamics using the fitted parameters and initial conditions
    sol = solve_and_plot_system(
        selected_params, 
        y0=y0_pre_peak, 
        t_span=t_span, 
        t_eval=t_eval, 
        system="cytokine",
        plot_all=False
    )