

# -*- coding: utf-8 -*-

## Assignment 4, Wout Geysen
## Last update 25/02/2025
## wout.geysen@student.uantwerpen.be

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from Models import dT_dt_Advanced_cytokine, dT_dt_Advanced_memory, dT_dt_Basic


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
    if system == "basic":
        def ode_system(t, y):
            return dT_dt_Basic(t, y, params)    
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