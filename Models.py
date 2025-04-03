### This script is where the main ODE models can be found
import numpy as np

basic_params = ["Tconv_suppress_base", "Tconv_prolif", "Tconv_decay", 
                "Treg_recruitment", "Treg_growth", "Treg_decay"]

intermediate_params = ["Tconv_suppress_base", "Tconv_prolif", "Tconv_decay",
                       "Treg_recruitment", "Treg_growth", "Treg_decay", 
                       "Mreg_growth", "Mreg_decay", "Mreg_conversion_base",
                       "K_reg", "tau"]

advanced_params = ["Tconv_suppress_base", "Tconv_prolif", "Tconv_decay",
                   "Treg_recruitment", "Treg_growth", "Treg_decay", 
                   "K_prolif", "K_suppress", "K_recruitment", "K_growth", "tau",
                   "IL2_production", "IL2_consumption"]

bounds = {
    "Tconv_suppress_base":  [0.014, 0.28],
    "Tconv_prolif":         [0.01, 0.30],   # 0.13-0.20        
    "Tconv_decay":          [0.05, 0.05],   # 0.05      
    "Treg_recruitment":     [0.012, 0.021], # 0.012-0.021
    "Treg_growth":          [0.001, 0.1],   # 
    "Treg_decay":           [0.03, 0.03],   # 0.03
    "Mreg_growth":          [0.01, 0.04],   # Almost stable with decay
    "Mreg_decay":           [0.02, 0.02],   #
    "Mreg_conversion_base": [0.01, 0.01],   # 0.01
    "K_reg":                [1., 1000.0],   #          
    "tau":                  [5.0, 5.0],     # 5.0
    "K_prolif":             [1., 40.],      # 10-30
    "K_suppress":           [1., 40.],      # 10-30
    "K_recruitment":        [1., 40.],      # 10-30
    "K_growth":             [1., 40.],      # 10-30
    "IL2_production":       [4.2, 6.5],     # 4.2-6.5
    "IL2_consumption":      [0.14, 0.23]    # 0.14-0.23
    }


def dT_dt_Basic(t, populations, param):
    Tconv_pop, Treg_pop, _, _ = populations

    dTconv_dt = (param["Tconv_prolif"] * Tconv_pop 
                    - param["Tconv_decay"] * Tconv_pop 
                    - param["Tconv_suppress_base"] * Tconv_pop * Treg_pop 
                    - param["Treg_recruitment"] * Tconv_pop 
                    )
    
    dTreg_dt = (param["Treg_growth"] * Treg_pop 
                    - param["Treg_decay"] * Treg_pop 
                    + param["Treg_recruitment"] * Tconv_pop 
                    )
    
    return np.array([dTconv_dt, dTreg_dt, 0 ,0])


def dT_dt_Advanced_memory(t, populations, param):
    # Reading out the current populations
    Tconv_pop, Treg_pop, _ , Mreg_pop = populations
    # Population dependent parameters:
    suppress_rate_Tconv = (param["Tconv_suppress_base"] 
                            * (1 - Treg_pop / (Treg_pop + param["K_reg"]))
                            * (1 - np.e**(-t/param["tau"]))
                            )
    conversion_rate_Mreg = param["Mreg_conversion_base"] * (1 - np.e**(-t/param["tau"]))
    

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


def dT_dt_Advanced_cytokine(t, populations, param):
    # Fix the unpacking syntax
    Tconv_pop, Treg_pop, IL2, _ = populations
    
    suppress_rate_Tconv = (param["Tconv_suppress_base"]
                           * (1 - np.exp(-t/param["tau"]))
                           )
    dTconv_dt = (param["Tconv_prolif"] * (IL2 / (IL2 + param["K_prolif"])) * Tconv_pop 
                    - param["Tconv_decay"] * Tconv_pop 
                    - suppress_rate_Tconv * (IL2 / (IL2 + param["K_suppress"])) * Tconv_pop * (Treg_pop) 
                    - param["Treg_recruitment"] * (IL2 / (IL2 + param["K_recruitment"])) * Tconv_pop)
    dTreg_dt = (param["Treg_growth"] * (IL2 / (IL2 + param["K_growth"])) * Treg_pop 
                    - param["Treg_decay"] * Treg_pop 
                    + param["Treg_recruitment"] * (IL2 / (IL2 + param["K_recruitment"])) * Tconv_pop)
    dIL2_dt = (param["IL2_production"] * Tconv_pop 
                    - param["IL2_consumption"] * (Tconv_pop + Treg_pop) *IL2)
    
    return np.array([dTconv_dt, dTreg_dt, dIL2_dt, 0])