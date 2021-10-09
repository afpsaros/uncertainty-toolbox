# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:12:30 2021

@author: afpsa
"""

import sys, os
sys.path.append(os.path.abspath('../'))
import uncertainty_toolbox as uct
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_scaled_residuals(mu, sigma, y):
    
    residuals = mu - y
    return residuals / sigma

def CRUDE_cdf(mu_recal, sigma_recal, y_recal, mu_eval, sigma_eval, y_evals):
    
    cdf_list = []
    for y in y_evals:
        cdf_list.append(np.mean(
            (y_recal - mu_recal) / sigma_recal <=
            (y - mu_eval) / sigma_eval))
    
    return np.array(cdf_list)

def get_stds_from_cdfs(means, un_cal_stds, inv_recal = {'method': None,
                                                        'model': None}):
    
    cdfs = []
    if inv_recal['method'] in [None, 'cdf_recal']:
        for mu, sigma in zip(means, un_cal_stds):
            norm = stats.norm(loc=mu, scale=sigma)
            cdfs.append(norm.cdf)
    elif inv_recal['method'] == 'CRUDE_recal': 
        for mu, sigma in zip(means, un_cal_stds):
            cdfs.append([mu, sigma])
    
    stds = []
    for cdf, mu in zip(cdfs, means):
        locs = np.linspace(0, 5, 1000)
        
        if inv_recal['method'] is None:
            cdf_vals = cdf(np.sqrt(locs)) - cdf(-np.sqrt(locs))
            
        elif inv_recal['method'] == 'cdf_recal':
            cdf_vals = inv_recal['model'].predict(cdf(np.sqrt(locs))) - inv_recal['model'].predict(cdf(-np.sqrt(locs)))
         
        elif inv_recal['method'] == 'CRUDE_recal':
            
            cdf_new = lambda y: CRUDE_cdf(*inv_recal['model'], 
                                      *cdf,
                                      y)
            cdf_vals = cdf_new(np.sqrt(locs)) - cdf_new(-np.sqrt(locs))
            
        # plt.plot(cdf_vals)
            
        vec = (1 - cdf_vals)  
        
        app = np.sqrt(np.trapz(vec, x = locs) - mu**2) if \
            np.trapz(vec, x = locs) - mu**2 > 0 else 0.01
        stds.append(app)
        
    return stds

def get_calibrator(mu, sigma, y, method):
    if method == 'std_recal':
        return uct.recalibration.get_std_recalibrator(mu, sigma, y)
    elif method == 'cdf_recal': 
        exp_props, obs_props = get_proportions(mu, sigma, y, num_bins = 200
                                               , prop_type = 'quantile')

        return uct.iso_recal(exp_props, obs_props) 
    elif method == 'CRUDE':
        return np.sort(get_scaled_residuals(mu, sigma, y)).tolist()
    
def get_proportions(mu, sigma, y, cal_dict = {'method': None, 'model': None},
                    num_bins = 20, prop_type = 'quantile'):
    
    if cal_dict['method'] is None:
        return uct.get_proportion_lists_vectorized(mu, sigma, y, num_bins = num_bins,
                                                   prop_type = prop_type)
            
    elif cal_dict['method'] == 'std_recal':
        return uct.get_proportion_lists_vectorized(mu, cal_dict['model'](sigma), y, 
                                                   num_bins = num_bins,
                                                   prop_type = prop_type)
    
    elif cal_dict['method'] in ['cdf_recal', 'CRUDE']:
        return uct.get_proportion_lists_vectorized(mu, sigma, y,
                                                   recal_model=cal_dict['model'], 
                                                   num_bins = num_bins,
                                                   prop_type = prop_type)      

def get_rmsce(props):

    squared_diff_proportions = np.square(props[0] - props[1])
    return np.sqrt(np.mean(squared_diff_proportions))
                    
    
def get_intervals(mu, sigma, p, cal_dict = {'method': None,
                                     'model': None}):
    
    if cal_dict['method'] is None:
        interval = uct.get_prediction_interval(mu, sigma, p)
            
    elif cal_dict['method'] == 'std_recal':
        interval = uct.get_prediction_interval(mu, cal_dict['model'](sigma), p)
    
    elif cal_dict['method'] in ['cdf_recal', 'CRUDE']:
        interval = uct.get_prediction_interval(mu, sigma, p,
                                           recal_model=cal_dict['model'])
    
    return [interval.lower, interval.upper]

    
    
        
        

