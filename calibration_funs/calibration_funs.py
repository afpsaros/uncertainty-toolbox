# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:12:30 2021

@author: afpsa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from metrics_calibration import *
from recalibration import *

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

def get_means_from_cdfs(un_cal_means, un_cal_stds, inv_recal = {'method': None,
                                                        'model': None},
                        locs = np.linspace(0, 10, 1000)):
    
    # Only for CDF and CRUDE recalibration
    if inv_recal['method'] == 'CRUDE_recal':
        crude_mean = np.mean(inv_recal['model'])
        
    means = []
    for mu, sigma in zip(un_cal_means, un_cal_stds):
        if inv_recal['method'] == 'cdf_recal':
            norm = stats.norm(loc=mu, scale=sigma)
            cdf = norm.cdf
            
            mu_vals_1 = 1 - inv_recal['model'].predict(cdf(locs))
            mu_vals_2 = inv_recal['model'].predict(cdf(-locs))
        
            means.append(np.trapz(mu_vals_1, x = locs) - np.trapz(mu_vals_2, x = locs))
            
        elif inv_recal['method'] == 'CRUDE_recal':
            means.append(mu + sigma * crude_mean)
            
    return means
    
def get_stds_from_cdfs(means, un_cal_stds, inv_recal = {'method': None,
                                                        'model': None},
                       locs = np.linspace(0, 10, 1000)):
    
    stds = []
    if inv_recal['method'] in [None, 'cdf_recal']:
        for mu, sigma in zip(means, un_cal_stds):
            norm = stats.norm(loc=mu, scale=sigma)
            cdf = norm.cdf
            
            if inv_recal['method'] is None:
                cdf_vals = cdf(np.sqrt(locs)) - cdf(-np.sqrt(locs))
            elif inv_recal['method'] == 'cdf_recal':
                cdf_vals = inv_recal['model'].predict(cdf(np.sqrt(locs))) - inv_recal['model'].predict(cdf(-np.sqrt(locs)))
                
            vec = (1 - cdf_vals)  
        
            app = np.sqrt(np.trapz(vec, x = locs) - mu**2) if \
                np.trapz(vec, x = locs) - mu**2 > 0 else 0.01
            stds.append(app)
                
    elif inv_recal['method'] == 'CRUDE_recal':
        crude_mean = np.mean(inv_recal['model'])
        crude_std = np.sqrt(np.mean((inv_recal['model'] - crude_mean)**2))
        
        for mu, sigma in zip(means, un_cal_stds):
            stds.append(sigma * crude_std)
    
    return stds

def get_calibrator(mu, sigma, y, method):
    if method == 'std_recal':
        return get_std_recalibrator(mu, sigma, y)
    elif method == 'cdf_recal': 
        exp_props, obs_props = get_proportions(mu, sigma, y, num_bins = 200
                                               , prop_type = 'quantile')

        return iso_recal(exp_props, obs_props) 
    elif method == 'CRUDE':
        return np.sort(get_scaled_residuals(mu, sigma, y)).tolist()
    
def get_proportions(mu, sigma, y, cal_dict = {'method': None, 'model': None},
                    num_bins = 20, prop_type = 'quantile'):
    
    if cal_dict['method'] is None:
        return get_proportion_lists_vectorized(mu, sigma, y, num_bins = num_bins,
                                                   prop_type = prop_type)
            
    elif cal_dict['method'] == 'std_recal':
        return get_proportion_lists_vectorized(mu, cal_dict['model'](sigma), y, 
                                                   num_bins = num_bins,
                                                   prop_type = prop_type)
    
    elif cal_dict['method'] in ['cdf_recal', 'CRUDE']:
        return get_proportion_lists_vectorized(mu, sigma, y,
                                                   recal_model=cal_dict['model'], 
                                                   num_bins = num_bins,
                                                   prop_type = prop_type)      

def get_rmsce(props):

    squared_diff_proportions = np.square(props[0] - props[1])
    return np.sqrt(np.mean(squared_diff_proportions))
                    
    
def get_intervals(mu, sigma, p, cal_dict = {'method': None,
                                     'model': None}):
    
    if cal_dict['method'] is None:
        interval = get_prediction_interval(mu, sigma, p)
            
    elif cal_dict['method'] == 'std_recal':
        interval = get_prediction_interval(mu, cal_dict['model'](sigma), p)
    
    elif cal_dict['method'] in ['cdf_recal', 'CRUDE']:
        interval = get_prediction_interval(mu, sigma, p,
                                           recal_model=cal_dict['model'])
    
    return [interval.lower, interval.upper]

    
    
        
        

