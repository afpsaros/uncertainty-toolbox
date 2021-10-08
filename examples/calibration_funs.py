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

def get_stds_from_cdfs(cdfs, means, inv_recal = {'method': None,
                                                 'model': None}):
    stds = []
    for cdf, mu in zip(cdfs, means):
        locs = np.linspace(0, 10, 1000)
        
        if inv_recal['method'] is None:
            cdf_vals = cdf(np.sqrt(locs)) - cdf(-np.sqrt(locs))
            
        elif inv_recal['method'] == 'cdf_recal':
            cdf_vals = inv_recal['model'].predict(cdf(np.sqrt(locs))) - inv_recal['model'].predict(cdf(-np.sqrt(locs)))
         
        elif inv_recal['method'] == 'CRUDE_recal':
            
            cdf_new = lambda y: CRUDE_cdf(*inv_recal['model'], 
                                      *cdf,
                                      y)
            cdf_vals = cdf_new(np.sqrt(locs)) - cdf_new(-np.sqrt(locs))
            
        vec = (1 - cdf_vals)  
        
        app = np.sqrt(np.trapz(vec, x = locs) - mu**2) if \
            np.trapz(vec, x = locs) - mu**2 > 0 else 0.001
        stds.append(app)
        
    return stds

def get_calibrator(mu, sigma, y, method):
    if method == 'std_recal':
        return uct.recalibration.get_std_recalibrator(mu, sigma, y)
    else:
        exp_props, obs_props = uct.get_proportion_lists_vectorized(
            mu, sigma, y
        )

        return uct.iso_recal(exp_props, obs_props) 
    
def get_proportions(mu, sigma, y, cal_dict = {'method': None,
                                          'model': None},
                    num_bins = 10):
    
    if cal_dict['method'] is None:
        return uct.get_proportion_lists_vectorized(mu, sigma, y, num_bins = num_bins)
            
    elif cal_dict['method'] == 'std_recal':
        return uct.get_proportion_lists_vectorized(mu, cal_dict['model'](sigma), y, 
                                                   num_bins = num_bins)
    
    elif cal_dict['method'] in ['cdf_recal', 'CRUDE']:
        return uct.get_proportion_lists_vectorized(mu, sigma, y,
                                                   recal_model=cal_dict['model'], 
                                                   num_bins = num_bins)      
    
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

def get_CRUDE_recalibrator(mu, sigma, y):
    
    return np.sort(get_scaled_residuals(mu, sigma, y)).tolist()
                                                   
    
    
        
        

