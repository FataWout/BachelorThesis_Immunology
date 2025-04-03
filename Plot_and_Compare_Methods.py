# This script is where all analysing functions for the models are written, for example the comparison between models, variable values, heatmaps,...

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

param = {
    "Tconv_suppress_base": 0.02,
    "Tconv_prolif": 0.12,
    "Tconv_decay": 0.05,
    "Treg_recruitment": 0.01,
    "Treg_growth": 0.08,
    "Treg_decay": 0.03,
    "K_prolif": 10.0,
    "K_suppress": 15.0,
    "K_recruitment": 12.0,
    "K_growth": 8.0,
    "IL2_production": 5.,
    "IL2_consumption": 0.2,
    "K_reg": 500.0,
    "tau": 5.0,
    "Mreg_conversion_base": 0.005,
    "Mreg_growth": 0.01,
    "Mreg_decay": 0.005
}

parameter_descriptions = {
    "Tconv_suppress_base": "Base suppression rate of Tconv cells by Treg cells.",
    "Tconv_prolif": "Proliferation rate of Tconv cells, describing how fast they divide.",
    "Tconv_decay": "Decay rate of Tconv cells, representing their natural death rate.",
    "Treg_recruitment": "Recruitment rate of Treg cells by Tconv cells.",
    "Treg_growth": "Growth rate of Treg cells, representing their proliferation rate.",
    "Treg_decay": "Decay rate of Treg cells, describing their natural death rate.",
    "K_prolif": "Proliferation constant influencing T cell growth dynamics.",
    "K_suppress": "Suppression constant defining Treg-mediated immune suppression.",
    "K_recruitment": "Recruitment constant affecting Treg cell attraction.",
    "K_growth": "Growth constant impacting cytokine-mediated cell expansion.",
    "IL2_production": "Rate at which IL-2 cytokines are produced.",
    "IL2_consumption": "Rate at which IL-2 cytokines are consumed by T cells.",
    "K_reg": "Regulatory constant affecting immune suppression and activation.",
    "tau": "Time scaling constant for immune system response delays.",
    "Mreg_conversion_base": "Base conversion rate of Treg cells into Mreg cells.",
    "Mreg_growth": "Growth rate of Mreg cells, determining their proliferation rate.",
    "Mreg_decay": "Decay rate of Mreg cells, representing their natural death rate."
}

def dT_dt_Advanced_cytokine_mod(t, populations):
    return dT_dt_Advanced_cytokine(t, populations, param)

def dT_dt_Advanced_memory_mod(t, populations):
    return dT_dt_Advanced_memory(t, populations, param)

def dT_dt_Basic_mod(t, populations):
    return dT_dt_Basic(t, populations, param)



def solve_odeint(equation_system = "Memory", initial_populations = [100, 10, 25, 2], timestamps=np.linspace(0, 100, 100)):   
    """Solves given equation system for given timestamps."""
    # Check which equation system was given to use and solve using that model        
    if equation_system == "Memory":
        return odeint(dT_dt_Advanced_memory_mod, initial_populations, timestamps, tfirst=True)
    elif equation_system == "Cytokine":
        return odeint(dT_dt_Advanced_cytokine_mod, initial_populations, timestamps, tfirst=True)


def get_variable_explanation(parameter_name):
    """Returns the explanation for a given parameter."""
    # Return explanation of parameter_name when the name is an existing one.
    return parameter_descriptions.get(parameter_name, "Explanation not found.")

# This definition will be used to make sure in future plots Treg and Tconv have the same color but for each alteration a different color.
def get_color(value, parameter_values, cmap = cm.brg):
    """
    Returns color code of normalized given value in comparison to list of given values.
    Parameters:
        value: value to be normalized and converted into color
        parameter_values: list of values (including value) to normalize
        cmap: Choose a colormap (e.g., viridis, plasma, etc.)
    """
    if len(parameter_values) == 1:
        return cmap(0)
    # Normalization of value so that we get assigned color
    min_val = np.min(parameter_values)
    max_val = np.max(parameter_values)
    normalized_value = (value - min_val) / (max_val - min_val)
    return cmap(normalized_value)
    


def compare_variable_values(parameter_name, parameter_values, param, 
                            initial_populations, timestamps=(0, 100, 100),
                            equation_systems=["Standard"],
                            plot_Mreg=True, plot_Cytokine=True,
                            saveplot=False, 
                            directory_path=r"C:\Users\woutg\Documents\Wout\Universiteit\Bachelor 3\Bachelorproef\Afbeeldingen\Assignment_1"):
    """Plots comparison between given parameter values in given equation systems.
    Parameters:
        parameter_name (str): name of parameter that gets altered.
        parameter_values (list of floats): all values of parameter_name to be compared.
        timestamps (numpy array): timestamps at which the systems should be solved.
        equation_systems (list of str): the equation systems in which variable should be compared.
        saveplot (bool): should the plots be saved as images? Yes -> True, No -> False.
        directory path (str): in case of saveplot = True, define a directory path where plots should be saved.
    """
    
    # Check for correct parameter_name, if nonexistent it raises an error.
    if parameter_name not in parameter_descriptions:
        raise AttributeError(f"{parameter_name} is not a valid parameter name and thus no valid variable to change in the mathematical model.")
    
    # When only one equation system is given as type str, convert it to a list of one element to generalize
    if isinstance(equation_systems, str):
        equation_systems = [equation_systems]
    
    indices = list(range(0, len(parameter_values)))  # Enumeration indices of all parameter_values
    
    # Initiating plot figure and adjusting white space
    fig, ax = plt.subplots(1, len(equation_systems), figsize=(14 + 6 * (len(equation_systems)-2), 6))
    fig.subplots_adjust(top=0.85)
    
    # When only one equation system is given, convert ax to a list of one element to generalize
    if len(equation_systems) == 1:
        ax = [ax]
    
    # Looping through all given equation_systems
    for i, equation_system in enumerate(equation_systems):
        # Save the original value of the parameter
        reference_value = param[parameter_name]
        
        # Solve the system with the original parameter value
        reference_solution = solve_odeint(equation_system, initial_populations, timestamps)
        
        # Plotting reference solutions
        if equation_system == "MemoryEffects" or equation_system == "Cytokine":
            if plot_Cytokine:
                ax[i].plot(timestamps, reference_solution[:, 2], label=f"Cytokine levels ({equation_system})", linestyle=(0, (1, 1)), color="grey", linewidth=2)
            if plot_Mreg:
                ax[i].plot(timestamps, reference_solution[:, 3], label=f"Reference Mreg ({reference_value})", linestyle="dashdot", color="grey", linewidth=1)
        ax[i].plot(timestamps, reference_solution[:, 0], label=f"Reference Tconv ({reference_value})", linestyle="solid", color="grey", linewidth=1)
        ax[i].plot(timestamps, reference_solution[:, 1], label=f"Reference Treg ({reference_value})", linestyle=(0, (5, 5)), color="grey", linewidth=1)
        
        # Solve for each altered value of the parameter
        for index, value in enumerate(parameter_values):
            # Set the parameter locally for this iteration (instead of using globals)
            param[parameter_name] = value
            
            altered_solution = solve_odeint(equation_system, initial_populations, timestamps)
            
            # Plotting altered solutions in the same color
            if equation_system == "Memory" or equation_system == "Cytokine":
                if plot_Cytokine:
                    ax[i].plot(timestamps, altered_solution[:, 2], label=f"Cytokine levels ({value})", linestyle=(0, (1, 1)), color=get_color(index, indices), linewidth=2)
                if plot_Mreg:
                    ax[i].plot(timestamps, altered_solution[:, 3], label=f"Mreg ({value})", linestyle="dashdot", color=get_color(index, indices), linewidth=1)
            ax[i].plot(timestamps, altered_solution[:, 0], label=f"Tconv ({value})", linestyle="solid", color=get_color(index, indices))
            ax[i].plot(timestamps, altered_solution[:, 1], label=f"Treg ({value})", linestyle=(0, (5, 5)), color=get_color(index, indices))
        
        # Customize the plot
        ax[i].set_title(f"{equation_system} model")
        ax[i].set_xlabel("Time (days)")
        ax[i].set_ylabel("Population")
        ax[i].grid()

    # Restore the original value of the parameter
    param[parameter_name] = reference_value

    # Add single legend to the whole figure, since the colors and lines are equal on each plot    
    max_handles_axis = max(ax, key=lambda a: len(a.get_legend_handles_labels()[0]))  # Compare by number of handles
    handles, labels = max_handles_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(parameter_values) + 1)
    
    plt.suptitle(f"Effect of varying {parameter_name}", fontsize=16, fontweight="bold")

    # Add a second line with explanation in a smaller font
    fig.text(0.5, 0.93, "(" + get_variable_explanation(parameter_name) + ")", ha='center', va='top', fontsize=10, color='black')        
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
        
    # Save the plots if saveplot is true
    if saveplot:
        file_path = f"{directory_path}\\d_{parameter_name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
    plt.show()
   

def compare_equation_systems(systems_to_compare, timestamps,
                            plot_Mreg = True, plot_Cytokine = True):
    """
    Compare the dynamics of Tconv and Treg populations for different systems of equations.
    
    Parameters:
        systems_to_compare (list): List of equation system names (e.g., "Standard", "Saturation",...).
        timestamps (numpy array): Time points for solving the ODEs.
    """
    indices = list(range(0,len(systems_to_compare))) # enumeration indices of all equation_systems
    # Initializing figure
    plt.figure(figsize=(10, 6))
        
    # If only one ODE  model is given, plotlines will be different colors.
    if len(systems_to_compare) == 1:
        solution = solve_odeint(systems_to_compare[0], timestamps=timestamps)
        if systems_to_compare[0] == "Memory" or systems_to_compare[0] == "Cytokine":
            if plot_Cytokine == True:    
                plt.plot(timestamps, solution[:, 2], label=f"Cytokine levels")
            if plot_Mreg == True:
                plt.plot(timestamps, solution[:, 3], label=f"Mreg")              

        plt.plot(timestamps, solution[:, 0], label=f"Tconv")
        plt.plot(timestamps, solution[:, 1], label=f"Treg")

        plt.title(f"Evolution of {systems_to_compare[0]} model through time")

        
        
        
    # In the case of more given systems we solve for all given equation_systems and plot the solutions in same color
    else:        
        for index, equation_system in enumerate(systems_to_compare):
            solution = solve_odeint(equation_system, timestamps=timestamps)

            if equation_system == "Memory" and plot_Mreg == True:
                plt.plot(timestamps, solution[:, 3], label=f"Mreg ({equation_system})", linestyle="--", color=get_color(index, indices), linewidth=1)
            if equation_system == "Cytokine" and plot_Cytokine == True:  
                 plt.plot(timestamps, solution[:, 2], label=f"Cytokine levels ({equation_system})", linestyle=(0, (1,1)), color=get_color(index, indices), linewidth=2)
                                     

            plt.plot(timestamps, solution[:, 0], label=f"Tconv ({equation_system})", linestyle="solid", color=get_color(index, indices), linewidth=2.5)
            plt.plot(timestamps, solution[:, 1], label=f"Treg ({equation_system})", linestyle=(0,(5,10)), color=get_color(index, indices))
        plt.title("Comparison of Different Mathematical Systems")

    # Customize plot
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
        

def stochastic_effects(equation_system, param, initial_populations, timestamps, number_of_systems=3, stochastic_vars=None,
                            plot_Mreg = True, plot_Cytokine = True):
    """
    Compare the dynamics of Tconv and Treg populations for different systems of equations,
    with optional stochastic changes to specific variables.

    Parameters:
        equation_system (str): Equation system name (e.g., "Standard", "Saturation",...).
        timestamps (numpy array): Time points for solving the ODEs.
        stochastic_vars (list): List of variable names to apply stochastic changes (e.g., ["Tconv_prolif"]).
    """

    # Default to no stochastic variables if not provided
    if stochastic_vars is None:
        stochastic_vars = []

    # Save original variable values
    original_values = param

    # Initialize the figure with GridSpec
    fig = plt.figure(figsize=(10, 6))

    if number_of_systems > 10:
        gs = gridspec.GridSpec(1, 1, height_ratios=[0.95])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Top half: Plot
    ax_plot = fig.add_subplot(gs[0])
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    indices = list(range(0, number_of_systems))
    for index in indices:
        # Apply stochastic changes
        if not index == 0:
            for var in stochastic_vars:
                if var in original_values:
                    param[var] = original_values[var] + np.random.normal(0, original_values[var] * 0.25) 
                        
        # Solve the system

        solution = solve_odeint(equation_system, initial_populations, timestamps)
        variable_values = [round(param[var], 5) for var in stochastic_vars]

        # Plot results

        ax_plot.plot(timestamps, solution[:, 0], label=f"Tconv {index} {variable_values}", linestyle="solid", color=get_color(index, indices))
        ax_plot.plot(timestamps, solution[:, 1], label=f"Treg {index}", linestyle=(0, (5, 10)), color=get_color(index, indices))

                
        if equation_system == "Memory" or equation_system == "Cytokine":
            if plot_Cytokine == True:    
                ax_plot.plot(timestamps, solution[:, 2], label=f"Cytokine levels {index}", linestyle=(0,(1,1)), color=get_color(index, indices), linewidth=2)
            if plot_Mreg == True:
                ax_plot.plot(timestamps, solution[:, 3], label=f"Mreg {index}", linestyle="dashdot", color=get_color(index, indices), linewidth=1)

        # Restore original variable values
        param = original_values

    # Plot customization
    ax_plot.set_title("Comparison of Different Mathematical Systems")
    ax_plot.set_xlabel("Time")
    ax_plot.set_ylabel("Population")
    ax_plot.grid()

    if number_of_systems <10:
        # Bottom half: Legend
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis("off")  # Turn off axes for the legend area
        handles, labels = ax_plot.get_legend_handles_labels()
        ax_legend.legend(
            handles,
            labels,
            loc="center",
            ncol=number_of_systems,
            frameon=False,
            prop={'size': 10},         # Font size
        )
        
        # Add the second line with explanation in a smaller font
        fig.text(0.5,0.22,  # Positioning the text (x, y)
                f"{stochastic_vars}",
                ha='center', va='top',
                fontsize=10, color='black')        
        plt.tight_layout()

    plt.show()
  

def heatmap(equation_system, param, initial_populations, timestamps=np.linspace(0, 10, 100), 
            variable_names = ["Tconv_prolif", "suppress_rate_Tconv"], range_var1 = [0.05, 0.15],
            range_var2 = [0.01, 0.04], nsteps = 10):
        
    # Check for correct parameter_name, if non existant it raises an error.
    for name in variable_names:
        if name not in parameter_descriptions:
                raise AttributeError(f"{name} is not a valid attribute of this class and thus no valid variable to change in the mathematical model.")
        # When only one equation system is given as type str, it converts it to a list of one element to generalize

    # Saving original values
    original_values = param
        
    # Empty matrices to fill with population values
    heat_matrix_conv = np.zeros((nsteps, nsteps))
    heat_matrix_reg = np.zeros((nsteps, nsteps))
        
    # Permute through every parameter combination
    for k in range(0, nsteps):
        param[variable_names[0]] = max(range_var1) - (max(range_var1)-min(range_var1))/(nsteps-1) * k
        for l in range(0, nsteps):
            param[variable_names[1]]= min(range_var2) + (max(range_var2)-min(range_var2))/(nsteps-1) * l
            solution =  solve_odeint(equation_system, initial_populations, timestamps)
            heat_matrix_conv[k,l] = solution[-1,0] + solution[-1, 1]
            heat_matrix_reg[k,l] = solution[-1,2] + solution[-1, 3]
                
    # Restore original variable values
    param = original_values 
                
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
    # Defining x- and y-ticks and -labels
    x_ticks = np.linspace(0, nsteps-1, 8 , endpoint = True)  # Custom x-tick positions
    y_ticks = np.linspace(0, nsteps-1, 8 , endpoint = True)  # Custom x-tick positions
    x_labels = np.round(np.linspace(range_var2[0], range_var2[1], 8, endpoint = True), 3)
    y_labels = np.round(np.linspace(range_var1[1], range_var1[0], 8, endpoint = True), 3)

    # Plot heatmap for the first matrix
    im1 = axes[0].imshow(heat_matrix_conv, cmap='viridis', aspect='auto')
    axes[0].set_title("Conventional T cell population")
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_xlabel(variable_names[1])
    axes[0].set_ylabel(variable_names[0])

    # Plot heatmap for the second matrix
    im2 = axes[1].imshow(heat_matrix_reg, cmap='viridis', aspect='auto')
    axes[1].set_title("Regulatory T cell population")
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(y_labels)
    axes[1].set_xlabel(variable_names[1])
    axes[1].set_ylabel(variable_names[0])
    fig.suptitle(f"Populations in function of {variable_names[0]} and {variable_names[1]}")
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    return heat_matrix_reg

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