# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:29:05 2021

@author: afpsa
"""

import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from sklearn.model_selection import train_test_split

from calibration_funs import *

from inverse_iso_recal import *

import scipy.stats as stats

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(10, 7)
fig.tight_layout(pad=3.0)

fig2, axs2 = plt.subplots(2, 3)
fig2.set_size_inches(10, 7)
fig2.tight_layout(pad=3.0)

np.random.seed(11)
# Generate synthetic predictive uncertainty results
n_tot = 500

f_tot, std_orig, y_tot, x_tot = uct.synthetic_sine_heteroscedastic(n_tot)
# std_tot = np.linspace(1.7, 1.5, n_tot) * std_orig
std_tot = np.linspace(0.7, 0.5, n_tot) * std_orig
Y_tot = np.concatenate((f_tot.reshape(-1,1), std_tot.reshape(-1,1), y_tot.reshape(-1,1)), axis = 1)

x_test, x_recal, Y_test, Y_recal = train_test_split(x_tot, Y_tot, test_size=0.01, random_state=42)
f_test, std_test, y_test = Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]
# f_recal, std_recal, y_recal = Y_recal[:, 0], Y_recal[:, 1], Y_recal[:, 2]
f_recal, std_recal, y_recal = f_tot, std_tot, y_tot

exp_props_orig, obs_props_orig = get_proportions(f_tot, std_orig, y_tot
                                                 # , cal_dict = {'method': 'CRUDE',
                                                 #               'model': get_CRUDE_recalibrator(f_tot, std_orig, y_tot)}
                                                 # , num_bins = 10
                                                 )

exp_props_mod, obs_props_mod = get_proportions(f_tot, std_tot, y_tot)

interval = get_intervals(f_tot, std_orig, 0.95)

rmsce = uct.root_mean_squared_calibration_error(
    f_tot, std_orig, y_tot, recal_model=None
)

axs[0, 0].plot(x_tot, y_tot, '.')
axs[0, 0].plot(x_tot, f_tot)
axs[0, 0].fill_between(
        x_tot,
        interval[0],
        interval[1],
        color="lightsteelblue",
        alpha=0.4,
    )
axs[0, 0].set_ylim([-3, 3])
axs[0, 0].title.set_text('Original \n E = ' + str(round(rmsce,3)) + ', S = ' + 
                         str(round(np.mean(interval[1] - interval[0]),3)))

xx = np.arange(-5, 5, 0.1)
axs2[0, 0].hist(get_scaled_residuals(f_tot, std_orig, y_tot), bins = 20, density = True)
axs2[0, 0].plot(xx, stats.norm.pdf(xx, 0, 1))
axs2[0, 0].set_xlim([-5, 5])

rmsce = uct.root_mean_squared_calibration_error(
    f_tot, std_tot, y_tot, recal_model=None
)

interval = get_intervals(f_tot, std_tot, 0.95)

axs[0, 1].plot(x_tot, y_tot, '.')
axs[0, 1].plot(x_tot, f_tot)
axs[0, 1].fill_between(
        x_tot,
        interval[0],
        interval[1],
        color="lightsteelblue",
        alpha=0.4,
    )
axs[0, 1].set_ylim([-3, 3])
axs[0, 1].title.set_text('Modified \n E = ' + str(round(rmsce,3)) + ', S = ' + 
                         str(round(np.mean(interval[1] - interval[0]),3)))

axs2[0, 1].hist(get_scaled_residuals(f_tot, std_tot, y_tot), bins = 20, density = True)
axs2[0, 1].plot(xx, stats.norm.pdf(xx, 0, 1))
axs2[0, 1].set_xlim([-5, 5])
#%%
std_recalibrator = get_calibrator(f_recal, std_recal, y_recal, 'std_recal')
std_new = std_recalibrator(std_tot)

rmsce = uct.root_mean_squared_calibration_error(
    f_tot, std_new, y_tot, recal_model=None
)

exp_props_r1, obs_props_r1 = get_proportions(f_tot, std_tot, y_tot, 
                                             cal_dict={'method': 'std_recal',
                                                       'model': std_recalibrator})

interval = get_intervals(f_tot, std_tot, 0.95, 
                                             cal_dict={'method': 'std_recal',
                                                       'model': std_recalibrator})

axs[1, 0].plot(x_tot, y_tot, '.')
axs[1, 0].plot(x_tot, f_tot)
axs[1, 0].fill_between(
        x_tot,
        interval[0],
        interval[1],
        color="lightsteelblue",
        alpha=0.4,
    )
axs[1, 0].set_ylim([-3, 3])
axs[1, 0].title.set_text('std recal \n E = ' + str(round(rmsce,3)) + ', S = ' + 
                         str(round(np.mean(interval[1] - interval[0]),3)))

axs2[1, 0].hist(get_scaled_residuals(f_tot, std_new, y_tot), bins = 20, density = True)
axs2[1, 0].plot(xx, stats.norm.pdf(xx, 0, 1))
axs2[1, 0].set_xlim([-5, 5])

recal_model = get_CRUDE_recalibrator(f_recal, std_recal, y_recal)

rmsce = uct.root_mean_squared_calibration_error(
    f_tot, std_tot, y_tot, recal_model=recal_model 
)

exp_props_r2, obs_props_r2 = get_proportions(f_tot, std_tot, y_tot, 
                                             cal_dict={'method': 'CRUDE',
                                                       'model': recal_model})

interval = get_intervals(f_tot, std_tot, 0.95, 
                                             cal_dict={'method': 'CRUDE',
                                                       'model': recal_model})

#%%
cdfs = []
for mu, sigma in zip(f_tot, std_tot):
    cdfs.append([mu, sigma])
     
stds_recal_2 = get_stds_from_cdfs(cdfs, f_tot, inv_recal = {'method': 'CRUDE_recal',
                                                 'model': [f_recal, std_recal, y_recal]})
#%%
axs[1, 1].plot(x_tot, y_tot, '.')
axs[1, 1].plot(x_tot, f_tot)
axs[1, 1].fill_between(
        x_tot,
        interval[0],
        interval[1],
        color="lightsteelblue",
        alpha=0.4,
    )
axs[1, 1].set_ylim([-3, 3])
axs[1, 1].title.set_text('CRUDE recal \n E = ' + str(round(rmsce,3)) + ', S = ' + 
                         str(round(np.mean(interval[1] - interval[0]),3)))

axs2[1, 1].hist(get_scaled_residuals(f_tot, stds_recal_2, y_tot), bins = 20, density = True)
axs2[1, 1].plot(xx, stats.norm.pdf(xx, 0, 1))
axs2[1, 1].set_xlim([-5, 5])

exp_props, obs_props = uct.get_proportion_lists_vectorized(
            f_recal, std_recal, y_recal, num_bins = 500
        )
recal_model = uct.iso_recal(exp_props, obs_props)
#%%
##############


# cdfs = []
# for mu, sigma in zip(f_tot, std_orig):
#     norm = stats.norm(loc=mu, scale=sigma)
#     cdfs.append(norm.cdf)

# stds = get_stds_from_cdfs(cdfs, f_tot)

# plt.plot(std_orig)   
# plt.plot(stds)
#################
#%%
rmsce = uct.root_mean_squared_calibration_error(
    f_tot, std_tot, y_tot, recal_model=recal_model 
)

exp_props_r3, obs_props_r3 = get_proportions(f_tot, std_tot, y_tot, 
                                             cal_dict={'method': 'cdf_recal',
                                                       'model': recal_model})

interval = get_intervals(f_tot, std_tot, 0.95, 
                                             cal_dict={'method': 'cdf_recal',
                                                       'model': recal_model})

inv_recal_model = inverse_iso_recal(exp_props, obs_props)
cdfs = []
for mu, sigma in zip(f_tot, std_tot):
    norm = stats.norm(loc=mu, scale=sigma)
    cdfs.append(norm.cdf)
    
stds_recal_3 = get_stds_from_cdfs(cdfs, f_tot, inv_recal = {'method': 'cdf_recal',
                                                            'model': inv_recal_model})

axs[1, 2].plot(x_tot, y_tot, '.')
axs[1, 2].plot(x_tot, f_tot)
axs[1, 2].fill_between(
        x_tot,
        interval[0],
        interval[1],
        color="lightsteelblue",
        alpha=0.4,
    )
axs[1, 2].set_ylim([-3, 3])
axs[1, 2].title.set_text('cdf recal \n E = ' + str(round(rmsce,3)) + ', S = ' + 
                         str(round(np.mean(interval[1] - interval[0]),3)))

axs2[1, 2].hist(get_scaled_residuals(f_tot, stds_recal_3, y_tot), bins = 20, density = True)
axs2[1, 2].plot(xx, stats.norm.pdf(xx, 0, 1))
axs2[1, 2].set_xlim([-5, 5])

# plt.savefig('intervals_dec.png', dpi = 300, bbox_inches="tight")
plt.show()

#%%
plt.plot([0, 1], [0, 1], "--", label='ideal')
plt.plot(exp_props_orig, obs_props_orig, label='original')
plt.plot(exp_props_mod, obs_props_mod, label='modified')
plt.plot(exp_props_r1, obs_props_r1, label='std recal')
plt.plot(exp_props_r2, obs_props_r2, label='CRUDE recal')
plt.plot(exp_props_r3, obs_props_r3, label='cdf recal')
plt.legend()
# plt.savefig('calibration_dec.png', dpi = 300, bbox_inches="tight")
plt.show()






