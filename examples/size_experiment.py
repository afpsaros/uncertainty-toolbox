# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:57:04 2021

@author: afpsa
"""

import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from sklearn.model_selection import train_test_split

from calibration_funs import *

n_tot = 400

np.random.seed(11)
f_tot, std_orig, y_tot, x_tot = uct.synthetic_sine_heteroscedastic(n_tot)
std_tot = np.linspace(1.7, 1.5, n_tot) * std_orig
Y_tot = np.concatenate((f_tot.reshape(-1,1), std_tot.reshape(-1,1), y_tot.reshape(-1,1)), axis = 1)

x_test, x_recal, Y_test, Y_recal = train_test_split(x_tot, Y_tot, test_size=0.5, random_state=42)

f_test, std_test, y_test = Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]

rmsce_orig = uct.root_mean_squared_calibration_error(
    f_tot, std_orig, y_tot, recal_model=None
)

rmsce_mod = uct.root_mean_squared_calibration_error(
    f_test, std_test, y_test, recal_model=None
)

std_list = []
CRUDE_list = []
cdf_list = []
t_list = np.arange(1, 100, 5)
for t in t_list:
    
    print(t)
    
    f_recal, std_recal, y_recal = Y_recal[:t, 0], Y_recal[:t, 1], Y_recal[:t, 2]
    
    std_recalibrator = get_calibrator(f_recal, std_recal, y_recal, 'std_recal')
    std_new = std_recalibrator(std_test)
    
    std_list.append(uct.root_mean_squared_calibration_error(
        f_test, std_new, y_test
    ))
#
    recal_model = get_CRUDE_recalibrator(f_recal, std_recal, y_recal)
    CRUDE_list.append(uct.root_mean_squared_calibration_error(
        f_test, std_test, y_test, recal_model=recal_model 
    ))    
#    
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
            f_recal, std_recal, y_recal
        )
    recal_model = uct.iso_recal(exp_props, obs_props)
    
    cdf_list.append(uct.root_mean_squared_calibration_error(
        f_test, std_test, y_test, recal_model=recal_model 
    ))
#%%
plt.plot([t_list[0], t_list[-1]], [rmsce_orig, rmsce_orig], label = 'original')
plt.plot([t_list[0], t_list[-1]], [rmsce_mod, rmsce_mod], label = 'modified')
plt.plot(t_list, std_list, '-o', label = 'std recal')
plt.plot(t_list, CRUDE_list, '-x', label = 'CRUDE recal')
plt.plot(t_list, cdf_list, '-+', label = 'cdf recal')
plt.ylim([0.0001, 0.25])
plt.legend()
plt.savefig('400_inc.png', dpi = 300, bbox_inches="tight")
plt.show()
    
    

