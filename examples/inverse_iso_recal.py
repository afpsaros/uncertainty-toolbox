# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:35:42 2021

@author: afpsa
"""

import sys, os
sys.path.append(os.path.abspath('../'))

from typing import Callable, Union
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
import uncertainty_toolbox as uct

def inverse_iso_recal(
    exp_props: np.ndarray,
    obs_props: np.ndarray,
) -> IsotonicRegression:
    """Recalibration algorithm based on isotonic regression.

    Fits and outputs an isotonic recalibration model that maps observed
    probabilities to expected probabilities. This mapping proivdes
    the necessary adjustments to produce better calibrated outputs.

    Args:
        exp_props: 1D array of expected probabilities (values must span [0, 1]).
        obs_props: 1D array of observed probabilities.

    Returns:
        An sklearn IsotonicRegression recalibration model.
    """
    # Flatten
    exp_props = exp_props.flatten()
    obs_props = obs_props.flatten()
    min_obs = np.min(obs_props)
    max_obs = np.max(obs_props)

    iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    # just need observed prop values between 0 and 1
    # problematic if min_obs_p > 0 and max_obs_p < 1
    if not (min_obs == 0.0) and (max_obs == 1.0):
        print("Obs props not ideal: from {} to {}".format(min_obs, max_obs))
        
    # print(exp_props)

    exp_0_idx = get_q_idx(exp_props, 0.0)
    exp_1_idx = get_q_idx(exp_props, 1.0)
    within_01 = obs_props[exp_0_idx : exp_1_idx + 1]

    beg_idx, end_idx = None, None
    # Handle beg_idx
    if exp_0_idx != 0:
        min_obs_below = np.min(obs_props[:exp_0_idx])
        min_obs_within = np.min(within_01)
        if min_obs_below < min_obs_within:
            i = exp_0_idx - 1
            while obs_props[i] > min_obs_below:
                i -= 1
            beg_idx = i
        elif np.sum((within_01 == min_obs_within).astype(float)) > 1:
            # multiple minima in within_01 ==> get last min idx
            i = exp_1_idx - 1
            while obs_props[i] > min_obs_within:
                i -= 1
            beg_idx = i
        elif np.sum((within_01 == min_obs_within).astype(float)) == 1:
            beg_idx = int(np.argmin(within_01) + exp_0_idx)
        else:
            raise RuntimeError("Inspect input arrays. Cannot set beginning index.")
    else:
        beg_idx = exp_0_idx

    # Handle end_idx
    if exp_1_idx < obs_props.shape[0] - 1:
        max_obs_above = np.max(obs_props[exp_1_idx + 1 :])
        max_obs_within = np.max(within_01)
        if max_obs_above > max_obs_within:
            i = exp_1_idx + 1
            while obs_props[i] < max_obs_above:
                i += 1
            end_idx = i + 1
        elif np.sum((within_01 == max_obs_within).astype(float)) > 1:
            # multiple minima in within_01 ==> get last min idx
            i = beg_idx
            while obs_props[i] < max_obs_within:
                i += 1
            end_idx = i + 1
        elif np.sum((within_01 == max_obs_within).astype(float)) == 1:
            end_idx = int(exp_0_idx + np.argmax(within_01) + 1)
        else:
            raise RuntimeError("Inspect input arrays. Cannot set ending index.")
    else:
        end_idx = exp_1_idx + 1

    if end_idx <= beg_idx:
        raise RuntimeError("Ending index before beginning index")

    filtered_obs_props = obs_props[beg_idx:end_idx]
    filtered_exp_props = exp_props[beg_idx:end_idx]

    try:
        iso_model = iso_model.fit(filtered_exp_props, filtered_obs_props)
    except Exception:
        raise RuntimeError("Failed to fit isotonic regression model")

    return iso_model

def get_q_idx(exp_props: np.ndarray, q: float) -> int:
    """Utility function which outputs the array index of an element.

    Gets the (approximate) index of a specified probability value, q, in the expected proportions array.
    Used as a utility function in isotonic regression recalibration.

    Args:
        exp_props: 1D array of expected probabilities.
        q: a specified probability float.

    Returns:
        An index which specifies the (approximate) index of q in exp_props
    """
    num_pts = exp_props.shape[0]
    target_idx = None
    for idx, x in enumerate(exp_props):
        if idx + 1 == num_pts:
            if round(q, 2) == round(float(exp_props[-1]), 2):
                target_idx = exp_props.shape[0] - 1
            break
        if x <= q < exp_props[idx + 1]:
            target_idx = idx
            break
    if target_idx is None:
        raise ValueError("q must be within exp_props")
    return target_idx