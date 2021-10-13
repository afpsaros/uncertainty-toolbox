# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:57:04 2021

@author: afpsa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data import *
from recalibration import *
from calibration_funs import *

n_tot = 400

np.random.seed(11)
f_tot, std_orig, y_tot, x_tot = synthetic_sine_heteroscedastic(n_tot)
modifier = np.linspace(0.7, 0.5, n_tot) **2
std_tot = modifier * std_orig
Y_tot = np.concatenate((f_tot.reshape(-1,1), std_tot.reshape(-1,1), y_tot.reshape(-1,1)), axis = 1)

x_test, x_recal, Y_test, Y_recal = train_test_split(x_tot, Y_tot, test_size=0.5, random_state=42)

f_test, std_test, y_test = Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]
#%%
props = get_proportions(f_tot, std_orig, y_tot)
rmsce_orig = get_rmsce(props)

props = get_proportions(f_tot, std_tot, y_tot)
rmsce_mod = get_rmsce(props)
#%%
error_list = [[], [], []]
calibrators = ['std_recal', 'CRUDE', 'cdf_recal']

t_list = np.arange(1, 10, 5)
for t in t_list:
    
    print(t)
    f_recal, std_recal, y_recal = Y_recal[:t, 0], Y_recal[:t, 1], Y_recal[:t, 2]
    
    for i, cal in enumerate(calibrators):
        recal_model = get_calibrator(f_recal, std_recal, y_recal, cal)
        props = get_proportions(f_test, std_test, y_test, cal_dict={'method': cal,
                                                                    'model': recal_model})
        
        error_list[i].append(get_rmsce(props))
        
#%%
plt.plot([t_list[0], t_list[-1]], [rmsce_orig, rmsce_orig], label = 'original')
plt.plot([t_list[0], t_list[-1]], [rmsce_mod, rmsce_mod], label = 'modified')
symbols = ['-o', '-x', '-+']
labels = ['std recal', 'CRUDE recal', 'cdf recal']

for i in range(len(calibrators)):
    plt.plot(t_list, error_list[i], symbols[i], label = labels[i])

plt.ylim([0.0001, 0.25])
plt.legend()
# plt.savefig('400_dec.png', dpi = 300, bbox_inches="tight")
plt.show()
    
    

