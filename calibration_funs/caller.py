# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:29:05 2021

@author: afpsa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats

from data import *
from recalibration import *
from calibration_funs import *
#%%
np.random.seed(11)
n_tot = 500

f_tot, std_orig, y_tot, x_tot = synthetic_sine_heteroscedastic(n_tot)
modifier = np.linspace(1.3, 1.5, n_tot) **2
std_tot = modifier * std_orig

Y_tot = np.concatenate((f_tot.reshape(-1,1), std_tot.reshape(-1,1), y_tot.reshape(-1,1)), axis = 1)

x_test, x_recal, Y_test, Y_recal = train_test_split(x_tot, Y_tot, test_size=0.05, random_state=42)
f_test, std_test, y_test = Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]
f_recal, std_recal, y_recal = Y_recal[:, 0], Y_recal[:, 1], Y_recal[:, 2]
#%%
exp_props, obs_props, intervals, rmsces, means_final, stds_final = [], [], [], [], [], []
perc = 0.95
#%%
props = get_proportions(f_tot, std_orig, y_tot)
exp_props.append(props[0])
obs_props.append(props[1])

intervals.append(get_intervals(f_tot, std_orig, perc))

rmsces.append(get_rmsce(props))

means_final.append(f_tot)
stds_final.append(std_orig)
#%%
props = get_proportions(f_tot, std_tot, y_tot)
exp_props.append(props[0])
obs_props.append(props[1])

intervals.append(get_intervals(f_tot, std_tot, perc))

rmsces.append(get_rmsce(props))

means_final.append(f_tot)
stds_final.append(std_tot)
#%%
recal_model = get_calibrator(f_recal, std_recal, y_recal, 'std_recal')
std_r1 = recal_model(std_tot)

props = get_proportions(f_tot, std_tot, y_tot, cal_dict={'method': 'std_recal',
                                                         'model': recal_model})
exp_props.append(props[0])
obs_props.append(props[1])

intervals.append(get_intervals(f_tot, std_tot, perc, cal_dict={'method': 'std_recal',
                                                               'model': recal_model}))

rmsces.append(get_rmsce(props))

means_final.append(f_tot)
stds_final.append(std_r1)
#%%
recal_model = get_calibrator(f_recal, std_recal, y_recal, 'CRUDE')
props = get_proportions(f_tot, std_tot, y_tot, cal_dict={'method': 'CRUDE',
                                                         'model': recal_model})
exp_props.append(props[0])
obs_props.append(props[1])

intervals.append(get_intervals(f_tot, std_tot, perc, cal_dict={'method': 'CRUDE',
                                                               'model': recal_model}))

rmsces.append(get_rmsce(props))

inv_recal_model = recal_model

new_means = get_means_from_cdfs(f_tot, std_tot, 
                                     inv_recal = {'method': 'CRUDE_recal',
                                                  'model': inv_recal_model})

means_final.append(new_means)

stds_final.append(get_stds_from_cdfs(new_means, std_tot, 
                                     inv_recal = {'method': 'CRUDE_recal',
                                                  'model': inv_recal_model}))
#%%
recal_model = get_calibrator(f_recal, std_recal, y_recal, 'cdf_recal')

props = get_proportions(f_tot, std_tot, y_tot, cal_dict={'method': 'cdf_recal',
                                                         'model': recal_model})
exp_props.append(props[0])
obs_props.append(props[1])

intervals.append(get_intervals(f_tot, std_tot, perc, cal_dict={'method': 'cdf_recal',
                                                               'model': recal_model}))

rmsces.append(get_rmsce(props))
 
iexp_props, iobs_props = get_proportions(f_recal, std_recal, y_recal
                                         , prop_type = 'cdf_quantile')

inv_recal_model = inverse_iso_recal(iexp_props, iobs_props)

new_means = get_means_from_cdfs(f_tot, std_tot, 
                                     inv_recal = {'method': 'cdf_recal',
                                                  'model': inv_recal_model})

means_final.append(new_means)
     
stds_final.append(get_stds_from_cdfs(f_tot, std_tot, 
                                     inv_recal = {'method': 'cdf_recal',
                                                  'model': inv_recal_model}))
#%%
labels = ['original', 'modified', 'std recal', 'CRUDE recal', 'cdf recal']

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(10, 7)
fig.tight_layout(pad=4.5)

for i, (p, r) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]):
    axs[p, r].plot(x_tot, y_tot, '.')
    axs[p, r].plot(x_tot, means_final[i])
    axs[p, r].fill_between(
            x_tot,
            intervals[i][0],
            intervals[i][1],
            color="lightsteelblue",
            alpha=0.4,
        )
    axs[p, r].set_xlabel('x')   
    axs[p, r].set_ylabel('y')
    axs[p, r].set_ylim([-3, 3])
    axs[p, r].title.set_text(labels[i] + '\n E = ' + str(round(rmsces[i],3)) + ', S = ' + 
                             str(round(np.mean(intervals[i][1] - intervals[i][0]),3)))
# plt.savefig('data_inc.png', dpi = 300, bbox_inches="tight")
plt.show()    
#%%
fig2, axs2 = plt.subplots(2, 3)
fig2.set_size_inches(10, 7)
fig2.tight_layout(pad=4.0)

xx = np.arange(-5, 5, 0.1)
for i, (p, r) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]):
    scaled_res = get_scaled_residuals(means_final[i], stds_final[i], y_tot)
    
    axs2[p, r].hist(scaled_res, bins = 20, density = True)
    axs2[p, r].plot(xx, stats.norm.pdf(xx, 0, 1))
    mu, std = stats.norm.fit(scaled_res) 
    axs2[p, r].plot(xx, stats.norm.pdf(xx, mu, std))
    axs2[p, r].set_xlim([-5, 5])
    axs2[p, r].set_ylim([0, 0.5])
    axs2[p, r].set_xlabel('scaled residual')   
    axs2[p, r].set_ylabel('PDF')
    axs2[p, r].title.set_text(labels[i])
# plt.savefig('hist_inc.png', dpi = 300, bbox_inches="tight")
plt.show()
#%%
for i in range(len(stds_final)):
    plt.plot(exp_props[i], obs_props[i], label=labels[i])
plt.plot([0, 1], [0, 1], "--", label='ideal')
plt.xlabel('expected proportions')
plt.ylabel('observed proportions')
plt.legend()
plt.title('calibration plots')
# plt.savefig('calibration_inc.png', dpi = 300, bbox_inches="tight")
plt.show()

for i in range(len(means_final)):
    plt.plot(x_tot, means_final[i], label=labels[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('mean')
plt.title('new means')
# plt.savefig('means_inc.png', dpi = 300, bbox_inches="tight")
plt.show()

# plt.plot(modifier, label = 'modifier')
for i in range(len(stds_final)):
    plt.plot(x_tot, stds_final[i], label=labels[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('std')
plt.title('new stds')
# plt.savefig('std_inc.png', dpi = 300, bbox_inches="tight")
plt.show()

# plt.plot(modifier, label = 'modifier')
for i in range(len(stds_final)):   
    plt.plot(x_tot, intervals[i][1] - intervals[i][0], label=labels[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('interval width')
plt.title('new 95% confidence intervals')
# plt.savefig('perc_inc.png', dpi = 300, bbox_inches="tight")
plt.show()

# plt.plot(modifier, label = 'modifier')
for i in range(len(stds_final)):   
    plt.plot(x_tot, (intervals[i][1] - intervals[i][0]) / stds_final[i], label=labels[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('interval width / std')
plt.title('non normality plots')
# plt.savefig('non_normal_inc.png', dpi = 300, bbox_inches="tight")
plt.show()





