# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:39:52 2021

@author: afpsa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from inverse_iso_recal import *
import scipy.stats as stats

import sys, os
sys.path.append(os.path.abspath('../'))
import uncertainty_toolbox as uct
from calibration_funs import *

np.random.seed(11)
n_tot = 100

f_list, std_list, std_mod_list, y_list, x_list = [], [], [], [], []
f_con, std_con, std_mod_con, y_con = np.array([]), np.array([]), np.array([]), np.array([])
f_recal, std_recal, y_recal = np.array([]), np.array([]), np.array([])

for iden in np.linspace(1, 11, 100):
    f, std, y, x = uct.synthetic_sine_heteroscedastic(n_tot, iden)
    modifier = np.linspace(1 + (iden - 7) / 20, 1 + (2 + iden) / 20, n_tot) **2
    plt.plot(x, modifier)
    plt.xlabel('x')
    plt.ylabel('multiplier')
    std_mod = modifier * std
    
    # print(sum(std_mod))
    
    f_list.append(f), std_list.append(std), std_mod_list.append(std_mod)
    y_list.append(y), x_list.append(x)
    
    f_con = np.concatenate((f_con, f)) 
    std_con = np.concatenate((std_con, std))
    std_mod_con = np.concatenate((std_mod_con, std_mod))
    y_con = np.concatenate((y_con, y))
    
    arg = np.random.randint(0, n_tot)
    
    f_recal = np.concatenate((f_recal, f[arg].reshape(-1)))
    std_recal = np.concatenate((std_recal, std_mod[arg].reshape(-1)))
    y_recal = np.concatenate((y_recal, y[arg].reshape(-1)))

plt.savefig('modifiers_2.png', dpi = 300, bbox_inches="tight")
plt.show()    
#%%
exp_props, obs_props, intervals, rmsces = [], [], [], []
perc = 0.95
#%%
new_mean_list = []
#%%
props = get_proportions(f_con, std_con, y_con)
exp_props.append(props[0])
obs_props.append(props[1])

new_mean_list.append(f_list)

intervals.append([get_intervals(el1, el2, perc) for el1, el2 in zip(f_list, std_list)])

rmsces.append(get_rmsce(props))

#%%
props = get_proportions(f_con, std_mod_con, y_con)
exp_props.append(props[0])
obs_props.append(props[1])

new_mean_list.append(f_list)

intervals.append([get_intervals(el1, el2, perc) for el1, el2 in zip(f_list, std_mod_list)])

rmsces.append(get_rmsce(props))
#%%
recal_model = get_calibrator(f_recal, std_recal, y_recal, 'std_recal')

props = get_proportions(f_con, std_mod_con, y_con, cal_dict={'method': 'std_recal',
                                                         'model': recal_model})

exp_props.append(props[0])
obs_props.append(props[1])

new_mean_list.append(f_list)

intervals.append([get_intervals(el1, el2, perc, cal_dict={'method': 'std_recal',
                                                               'model': recal_model})
                   for el1, el2 in zip(f_list, std_mod_list)])

rmsces.append(get_rmsce(props))
#%%
recal_model = get_calibrator(f_recal, std_recal, y_recal, 'CRUDE')

props = get_proportions(f_con, std_mod_con, y_con, cal_dict={'method': 'CRUDE',
                                                         'model': recal_model})
exp_props.append(props[0])
obs_props.append(props[1])

inv_recal_model = recal_model
new_mean_list.append([get_means_from_cdfs(el1, el2, 
                                     inv_recal = {'method': 'CRUDE_recal',
                                                  'model': inv_recal_model})
                      for el1, el2 in zip(f_list, std_mod_list)])

intervals.append([get_intervals(el1, el2, perc, cal_dict={'method': 'CRUDE',
                                                               'model': recal_model})
                   for el1, el2 in zip(f_list, std_mod_list)])

rmsces.append(get_rmsce(props))
#%%
recal_model = get_calibrator(f_recal, std_recal, y_recal, 'cdf_recal')

props = get_proportions(f_con, std_mod_con, y_con, cal_dict={'method': 'cdf_recal',
                                                         'model': recal_model})
exp_props.append(props[0])
obs_props.append(props[1])

iexp_props, iobs_props = get_proportions(f_recal, std_recal, y_recal
                                         , prop_type = 'cdf_quantile')
inv_recal_model = inverse_iso_recal(iexp_props, iobs_props)

new_mean_list.append([get_means_from_cdfs(el1, el2, 
                                     inv_recal = {'method': 'cdf_recal',
                                                  'model': inv_recal_model})
                      for el1, el2 in zip(f_list, std_mod_list)])

intervals.append([get_intervals(el1, el2, perc, cal_dict={'method': 'cdf_recal',
                                                               'model': recal_model})
                   for el1, el2 in zip(f_list, std_mod_list)])

rmsces.append(get_rmsce(props))

print(rmsces)
#%%
idens = [0, 20, 50, 80]
cal_titles = ['original', 'modified', 'calibrated']

for iden in idens:
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(10, 4)
    fig.tight_layout(pad=3.0)
    
    for i, cal_type in enumerate([0, 1, 2]):

        axs[cal_type].plot(x, f_list[iden])
        axs[cal_type].plot(x, y_list[iden], 'o')
        axs[cal_type].fill_between(
                    x,
                    intervals[cal_type][iden][0],
                    intervals[cal_type][iden][1],
                    color="lightsteelblue",
                    alpha=0.4,
                )
        axs[cal_type].set_ylim([-4, 4])
        axs[cal_type].set_xlabel('x')
        axs[cal_type].set_ylabel('y')
        axs[cal_type].set_title(cal_titles[i] + ' - E = ' + str(round(rmsces[i], 3)))
    
    plt.savefig('data_don' + str(iden) + '.png', dpi = 300, bbox_inches="tight")
    plt.show()
#%%    
for i in range(3):
    plt.plot(exp_props[i], obs_props[i], label=cal_titles[i])
plt.plot([0, 1], [0, 1], "--", label='ideal')
plt.xlabel('expected percentage')
plt.ylabel('observed percentage')
plt.legend()
plt.savefig('calibration_don.png', dpi = 300, bbox_inches="tight")
plt.show()


    
    
    
