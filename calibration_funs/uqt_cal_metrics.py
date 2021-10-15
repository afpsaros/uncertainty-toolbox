"""
The following code has ben adopted by https://github.com/uncertainty-toolbox/uncertainty-toolbox
It has been adapted to the needs of our problems
"""

from typing import Any, Tuple, Optional
from argparse import Namespace
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy import stats
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from sklearn.isotonic import IsotonicRegression

def get_proportion_lists_vectorized(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 10,
    recal_model: Any = None,
    prop_type: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray]:
    """Arrays of expected and observed proportions

    Returns the expected proportions and observed proportion of points falling into
    intervals corresponding to a range of quantiles.
    Computations here are vectorized for faster execution, but this function is
    not suited when there are memory constraints.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        recal_model: an sklearn isotonoic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.

    Returns:
        A tuple of two numpy arrays, expected proportions and observed proportions

    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile", 'cdf_quantile']

    # Compute proportions
    if prop_type == 'cdf_quantile':
        exp_proportions = np.sort(np.array([0.] + 
                                            [stats.norm(loc=mu, scale=sigma).cdf(y) 
                                            for mu,sigma,y in
                                            zip(y_pred, y_std, y_true)] +[1.]))  
    else:
        exp_proportions = np.linspace(0, 1, num_bins)
       
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None and type(recal_model) is not list:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    norm = stats.norm(loc=0, scale=1)
    
    if prop_type == "interval":
        needed_lower_bound = 0.5 - in_exp_proportions / 2.0
        needed_upper_bound = 0.5 + in_exp_proportions / 2.0
        
        if type(recal_model) is not list:
            lower_bound = norm.ppf(needed_lower_bound)
            upper_bound = norm.ppf(needed_upper_bound)
        else:
            args_lower = [int(el * len(recal_model)) 
                          if int(el * len(recal_model)) < len(recal_model) 
                          else len(recal_model) - 1 for el in needed_lower_bound] 
            args_upper = [int(el * len(recal_model)) 
                          if int(el * len(recal_model)) < len(recal_model) 
                          else len(recal_model) - 1 for el in needed_upper_bound] 
            
            lower_bound = [recal_model[el] for el in args_lower]
            upper_bound = [recal_model[el] for el in args_upper]
            
        above_lower = normalized_residuals >= lower_bound
        below_upper = normalized_residuals <= upper_bound

        within_quantile = above_lower * below_upper
        obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)
    elif prop_type in ["quantile", 'cdf_quantile']:
        if type(recal_model) is not list:
            
            quantile_bound = norm.ppf(in_exp_proportions)
        else:

            args = [int(el * len(recal_model)) 
                          if int(el * len(recal_model)) < len(recal_model) 
                          else len(recal_model) - 1 for el in in_exp_proportions] 
            
            quantile_bound = [recal_model[el] for el in args]
            
        below_quantile = normalized_residuals <= quantile_bound
        obs_proportions = np.sum(below_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions

def get_prediction_interval(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    quantile: np.ndarray,
    recal_model: Optional[IsotonicRegression] = None,
) -> Namespace:
    """Return the centered predictional interval corresponding to a quantile.

    For a specified quantile level q (must be a float, or a singleton),
    return the centered prediction interval corresponding
    to the pair of quantiles at levels (0.5-q/2) and (0.5+q/2),
    i.e. interval that has nominal coverage equal to q.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        quantile: The quantile level to check.
        recal_model: A recalibration model to apply before computing the interval.

    Returns:
        Namespace containing the lower and upper bound corresponding to the
        centered interval.
    """

    if isinstance(quantile, float):
        quantile = np.array([quantile])

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std)
    assert_is_flat_same_shape(quantile)
    assert quantile.size == 1
    # Check that input std is positive
    assert_is_positive(y_std)

    if not np.logical_and((0.0 < quantile.item()), (quantile.item() < 1.0)):
        raise ValueError("Quantile must be greater than 0.0 and less than 1.0")

    # if recal_model is not None, calculate recalibrated quantile
    if recal_model is not None and type(recal_model) is not list:
        quantile = recal_model.predict(quantile)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=y_pred, scale=y_std)
    
    needed_lower_bound = 0.5 - quantile / 2.0
    needed_upper_bound = 0.5 + quantile / 2.0
    
    if type(recal_model) is not list:
        
        lower_bound = norm.ppf(needed_lower_bound)
        upper_bound = norm.ppf(needed_upper_bound)

    else:
        arg_lower = int(needed_lower_bound * len(recal_model)) \
                      if int(needed_lower_bound * len(recal_model)) < len(recal_model) \
                      else len(recal_model) - 1
                      
        arg_upper = int(needed_upper_bound * len(recal_model)) \
                      if int(needed_upper_bound * len(recal_model)) < len(recal_model) \
                      else len(recal_model) - 1
        
        lower_bound = y_pred + recal_model[arg_lower] * y_std
        upper_bound = y_pred + recal_model[arg_upper] * y_std

    bounds = Namespace(
        upper=upper_bound,
        lower=lower_bound,
    )

    return bounds
#%%
def root_mean_squared_calibration_error(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    vectorized: bool = False,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> float:
    """Root mean squared calibration error.
    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        vectorized: whether to vectorize computation for observed proportions.
                    (while setting to True is faster, it has much higher memory requirements
                    and may fail to run for larger datasets).
        recal_model: an sklearn isotonoic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
    Returns:
        A single scalar which calculates the root mean squared calibration error.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )

    squared_diff_proportions = np.square(exp_proportions - obs_proportions)
    rmsce = np.sqrt(np.mean(squared_diff_proportions))

    return rmsce

def get_proportion_lists(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 100,
    recal_model: IsotonicRegression = None,
    prop_type: str = "interval",
) -> Tuple[np.ndarray, np.ndarray]:
    """Arrays of expected and observed proportions
    Return arrays of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    Computations here are not vectorized, in case there are memory constraints.
    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        num_bins: number of discretizations for the probability space [0, 1].
        recal_model: an sklearn isotonoic regression model which recalibrates the predictions.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
    Returns:
        A tuple of two numpy arrays, expected proportions and observed proportions
    """
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)
    # Check that prop_type is one of 'interval' or 'quantile'
    assert prop_type in ["interval", "quantile"]

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    if prop_type == "interval":
        obs_proportions = [
            get_proportion_in_interval(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]
    elif prop_type == "quantile":
        obs_proportions = [
            get_proportion_under_quantile(y_pred, y_std, y_true, quantile)
            for quantile in in_exp_proportions
        ]

    return exp_proportions, obs_proportions

def get_proportion_in_interval(
    y_pred: np.ndarray, y_std: np.ndarray, y_true: np.ndarray, quantile: float
) -> float:
    """For a specified quantile, return the proportion of points falling into
    an interval corresponding to that quantile.
    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        quantile: a specified quantile level
    Returns:
        A single scalar which is the proportion of the true labels falling into the
        prediction interval for the specified quantile.
    """

    # Check that input arrays are flat
    assert_is_flat_same_shape(y_pred, y_std, y_true)
    # Check that input std is positive
    assert_is_positive(y_std)

    # Computer lower and upper bound for quantile
    norm = stats.norm(loc=0, scale=1)
    lower_bound = norm.ppf(0.5 - quantile / 2)
    upper_bound = norm.ppf(0.5 + quantile / 2)

    # Compute proportion of normalized residuals within lower to upper bound
    residuals = y_pred - y_true

    normalized_residuals = residuals.reshape(-1) / y_std.reshape(-1)

    num_within_quantile = 0
    for resid in normalized_residuals:
        if lower_bound <= resid and resid <= upper_bound:
            num_within_quantile += 1.0
    proportion = num_within_quantile / len(residuals)

    return proportion
#%%
from typing import Any, NoReturn, Union

def assert_is_flat_same_shape(*args: Any) -> Union[bool, NoReturn]:
    """Check if inputs are all same-length 1d numpy.ndarray.

    Args:
        args: the numpy arrays to check.

    Returns:
        True if all arrays are flat and the same shape, or else raises assertion error.
    """

    assert isinstance(args[0], np.ndarray), "All inputs must be of type numpy.ndarray"
    first_shape = args[0].shape
    for arr in args:
        assert isinstance(arr, np.ndarray), "All inputs must be of type numpy.ndarray"
        assert len(arr.shape) == 1, "All inputs must be 1d numpy.ndarray"
        assert arr.shape == first_shape, "All inputs must have the same length"

    return True


def assert_is_positive(*args: Any) -> Union[bool, NoReturn]:
    """Assert that all numpy arrays are positive.

    Args:
        args: the numpy arrays to check.

    Returns:
        True if all elements in all arrays are positive values, or else raises assertion error.
    """
    for arr in args:
        assert isinstance(arr, np.ndarray), "All inputs must be of type numpy.ndarray"
        assert all(arr > 0.0)

    return True
