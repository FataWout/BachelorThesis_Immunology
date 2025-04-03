
# -*- coding: utf-8 -*-

## Assignment 4, Wout Geysen
## Last update 25/02/2025
## wout.geysen@student.uantwerpen.be
import numpy as np
from Models import bounds

def round_sig(value, sig=4):
    """Round a number to a specified number of significant figures."""
    if value == 0:
        return 0  # Avoid log issues with zero
    return round(value, sig - int(np.floor(np.log10(abs(value)))) - 1)

def compute_means(param_list, bounds_dict=bounds):
    mean_dict = {}
    for param in param_list:
        value = bounds_dict[param]
        if isinstance(value, list):  # If it's a list, calculate the mean
            mean_dict[param] = np.mean(value)
        else:  # If it's a single value, just assign it
            mean_dict[param] = value
    return mean_dict

def filter_bounds(param_list, bounds_dict=bounds):
    """Filters the bounds dictionary to include only parameters in param_list."""
    return {param: bounds_dict[param] for param in param_list if param in bounds_dict}

# Fitting all parameters except decay rates, these are proportional to growth rates
def restructure_bounds(param_list, bounds_dict=bounds, epsilon=1e-8):
    """Restructures bounds for SciPy's least_squares function with strict inequality."""
    lower_bounds = []
    upper_bounds = []
    
    for param in param_list:
        value = bounds_dict[param]
        if isinstance(value, list):  # If range is given, extract lower and upper bounds
            lower, upper = value
        else:  # If single value, set lower and upper (but ensure strict inequality)
            lower, upper = value, value
        
        # Ensure strict inequality by adding a tiny epsilon if necessary
        if lower == upper:
            upper += epsilon  # Slightly increase upper bound
        
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    
    return (np.array(lower_bounds), np.array(upper_bounds))


