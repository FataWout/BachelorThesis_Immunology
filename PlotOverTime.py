

# -*- coding: utf-8 -*-

## Assignment 4, Wout Geysen
## Last update 25/02/2025
## wout.geysen@student.uantwerpen.be

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def dT_dt_Advanced_memory(t, populations, param):
    # Reading out the current populations
    Tconv_pop, Treg_pop, _ , Mreg_pop = populations
    # Population dependent parameters:
    suppress_rate_Tconv = param["Tconv_suppress_base"] * (1 - Treg_pop / (Treg_pop + param["K_reg"])) * (1 - np.e**(-t/param["tau"]))
    conversion_rate_Mreg = param["Mreg_conversion_base"] * (t / (t +  param["tau"]))
    

    dTconv_dt = (param["Tconv_prolif"] * Tconv_pop 
                    - param["Tconv_decay"] * Tconv_pop 
                    - suppress_rate_Tconv * Tconv_pop * (Treg_pop + Mreg_pop) 
                    - param["Treg_recruitment"] * Tconv_pop 
                    )
    dTreg_dt = (param["Treg_growth"] * Treg_pop 
                    - param["Treg_decay"] * Treg_pop 
                    + param["Treg_recruitment"] * Tconv_pop 
                    - conversion_rate_Mreg * Treg_pop
                    )
    dMreg_dt = (conversion_rate_Mreg * (Treg_pop)
                    + param["Mreg_growth"] * Mreg_pop
                    - param["Mreg_decay"] * Mreg_pop
                    )
    return np.array([dTconv_dt, dTreg_dt, 0, dMreg_dt])


def dT_dt_Advanced_cytokine(t, populations, params):
    # Fix the unpacking syntax
    Tconv_pop, Treg_pop, IL2, _ = populations
    
    suppress_rate_Tconv = params["Tconv_suppress_base"]
                           #* (1 - Treg_pop / (Treg_pop + param["K_reg"])) 
                           #                                  * (1 - np.exp(-t/param["tau"])))
    
    dTconv_dt = (params["Tconv_prolif"] * (IL2 / (IL2 + params["K_prolif"])) * Tconv_pop 
                    - params["Tconv_decay"] * Tconv_pop 
                    - suppress_rate_Tconv * (IL2 / (IL2 + params["K_suppress"])) * Tconv_pop * (Treg_pop) 
                    - params["Treg_recruitment"] * (IL2 / (IL2 + params["K_recruitment"])) * Tconv_pop)
    dTreg_dt = (params["Treg_growth"] * (IL2 / (IL2 + params["K_growth"])) * Treg_pop 
                    - params["Treg_decay"] * Treg_pop 
                    + params["Treg_recruitment"] * (IL2 / (IL2 + params["K_recruitment"])) * Tconv_pop)
    dIL2_dt = (params["IL2_production"] * Tconv_pop 
                    - params["IL2_consumption"] * (Tconv_pop + Treg_pop) * IL2)
    
    return np.array([dTconv_dt, dTreg_dt, dIL2_dt, 0])

def solve_and_plot_system(params, y0, t_span, t_eval=None, system = "memory"):
    """
    Solves the ODE system and plots population dynamics.

    Args:
        param: Dictionary of parameters.
        y0: List of initial conditions [Tconv, Treg, IL2, Mreg].
        t_span: Tuple for the time span (start, end).
        t_eval: Array of time points at which to store the computed solutions.

    Returns:
        Solution object from solve_ivp.
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)  # Default time resolution

    # Define ODE system with parameters
    if system == "memory":
        def ode_system(t, y):
            return dT_dt_Advanced_memory(t, y, params)
    elif system == "cytokine":
        def ode_system(t, y):
            return dT_dt_Advanced_cytokine(t, y, params)       

    # Solve ODE system
    sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], label='Tconv Population', color='tab:blue')
    plt.plot(sol.t, sol.y[1], label='Treg Population', color='tab:orange')
    if system == "memory":
        plt.plot(sol.t, sol.y[3], label='Mreg Population', color='tab:green')
    elif system == "cytokine":
        plt.plot(sol.t, sol.y[2], label='IL-2 Concentration', color='tab:green')
    
    plt.xlabel('Time')
    plt.ylabel('Population / Concentration')
    plt.title('Population Dynamics Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return sol