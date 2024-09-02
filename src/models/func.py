# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:17:57 2023

@author: Janathan

This script reads a training CSV, outputs predictions for the data, and applies
the score metric to compute the final score. 


"""

# 'F5' Começa a debuger o codigo 
# 'F10' Analisar a linha sem entrar no codigo 
# 'F11' Analisar linha  e entrar no codigo 
# 'SHIFT-F11' sair do bloco de codigo atual e continuar a execução


import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

X_SIZE = 5      # 5 input samples
N_GPS = 2       # 2 GPSs (unit1 and unit2)
N_GPS_COORD = 2 # 2 GPS coords (latitude & longitude)
N_ARR = 4       # 4 arrays
N_BEAMS = 64    # 64 beams per array
IDX_COL1 = 'unique_index' # index col in the training CSV
IDX_COL2 = 'abs_index'    # index col in the CSVs of the V2V dataset 
                          # this indices are the same, just dif names

def norm_2pi(x):
    """
    Normalize angles in radians to the range of -pi to pi.

    Parameters:
    - x (numpy.ndarray or array-like): Input angles in radians.

    Returns:
    - numpy.ndarray: Angles normalized to the range of -pi to pi.

    The function takes an array of angles in radians and normalizes them to the
    range of -pi to pi. It handles values outside this range by applying modular
    arithmetic to bring them within the specified range.

    Examples:
    >>> import numpy as np
    >>> angles = np.array([-3*np.pi, 2*np.pi, np.pi/2])
    >>> norm_2pi(angles)
    array([ 3.14159265, -3.14159265,  1.57079633])

    Note:
    The function uses modular arithmetic to ensure the normalized angles lie
    within the range of -pi to pi. If an angle is less than -pi after
    normalization, a warning message is printed.
    """
    # -pi to pi
    x_normed = np.empty_like(x)
    x_normed[:] = x
    for i in range(len(x)):
        if abs(x[i]) >= np.pi:
            x_normed[i] = x[i] % (2*np.pi)

            if x[i] >= np.pi:
                x_normed[i] -= 2*np.pi

        while x_normed[i] < -np.pi:
            x_normed[i] += 2*np.pi

        while x_normed[i] > np.pi:
            x_normed[i] -= 2*np.pi

        if x_normed[i] < -np.pi:
            print(f'{i}, {x_normed[i]}')

    return x_normed


def compute_ori_from_pos_delta(lat_deltas, lon_deltas):
    """
    Compute orientation in the range [-pi, pi].

    Parameters:
    - lat_deltas (numpy.ndarray or array-like): Differences in latitudes.
    - lon_deltas (numpy.ndarray or array-like): Differences in longitudes.Orient

    Returns:
    - numpy.ndarray: Orientations corresponding to the differences in latitudes and longitudes.

    The function computes orientation based on differences in latitudes and longitudes.
    The orientation is given in radians and falls within the range of -pi to pi.
    The orientation is calculated using arctan(delta_lat / delta_lon).
    If delta_lon is 0, the orientation is determined based on the sign of delta_lat.

    Thresholds:
    If delta_lat is below a certain threshold (thres_lat), it is considered as 0.

    If lat_deltas and lon_deltas are N x 2, the function computes the difference
    between the two columns.
    If lat_deltas and lon_deltas are N x 1, it computes differences in consecutive samples
    (uses two positions at different times to get orientation).


    Examples:
    >>> import numpy as np
    >>> lat_deltas = np.array([0, 1, -1, 0])
    >>> lon_deltas = np.array([1, 0, 0, -1])
    >>> compute_ori_from_pos_delta(lat_deltas, lon_deltas)
    array([ 0.        ,  1.57079633, -1.57079633,  3.14159265])

    """
    n_samples = len(lat_deltas)
    pose = np.zeros(n_samples)

    for i in range(n_samples):

        delta_lat = lat_deltas[i-1]
        delta_lon = lon_deltas[i-1]

        if delta_lon == 0:
            if delta_lat == 0:
                pose[i] = 0
                continue
            elif delta_lat > 0:
                slope = np.pi / 2
            elif delta_lat < 0:
                slope = -np.pi / 2
        else:
            slope = np.arctan(delta_lat / delta_lon)
            if delta_lat == 0:
                slope = np.pi if delta_lon < 0 else 0
            elif delta_lat < 0 and delta_lon < 0:
                slope = -np.pi + slope
            elif delta_lon < 0 and delta_lat > 0:
                slope = np.pi + slope

        pose[i] = slope

    return pose


def estimate_positions(input_positions, delta_input, delta_output):
    """
    Estimate positions based on input positions using linear interpolation.

    Parameters:
    - input_positions (numpy.ndarray): Input positions with shape (n_samples, n_points, 2).
    - delta_input (float): Time difference between consecutive input samples.
    - delta_output (float): Time difference for the desired output position.

    Returns:
    - numpy.ndarray: Estimated output positions with shape (n_samples, 2).

    The function estimates output positions based on input positions using linear interpolation.
    It assumes that input positions are provided at regular intervals with a specified time difference (delta_input).
    The output positions are estimated at a future time (delta_output) using linear interpolation.

    Examples:
    >>> input_positions = np.array([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    ...                              [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]])
    >>> delta_input = 1.0
    >>> delta_output = 2.0
    >>> estimate_positions(input_positions, delta_input, delta_output)
    array([[ 5.,  6.],
           [10., 11.]])

    Note:
    - The input_positions array should have shape (n_samples, n_points, 2), where n_samples is the number of samples,
      n_points is the number of input points, and the last dimension represents the (latitude, longitude) coordinates.
    - The function uses linear interpolation to estimate the positions at a future time (delta_output).
    """

    # Calculate the number of samples in the input positions array
    n_samples = input_positions.shape[0]

    # Initialize an array to store the estimated output positions
    out_pos = np.zeros((n_samples, 2))

    # Determine the size of each input position array (number of samples)
    x_size = input_positions.shape[1]

    # Define the time points corresponding to each sample in the input positions array
    x = delta_input * np.arange(x_size)

    # Iterate over each sample to estimate the corresponding output position
    for sample_idx in tqdm(range(n_samples), desc='Estimating input positions'):
    # Extract the input positions for the current sample
        input_pos = input_positions[sample_idx]

        # Perform linear interpolation to estimate latitude and longitude at the output time
        f_lat = scipy.interpolate.interp1d(x, input_pos[:, 0], fill_value='extrapolate')
        f_lon = scipy.interpolate.interp1d(x, input_pos[:, 1], fill_value='extrapolate')

        # Calculate the estimated latitude and longitude at the output time
        out_pos[sample_idx, 0] = f_lat(x[-1] + delta_output)
        out_pos[sample_idx, 1] = f_lon(x[-1] + delta_output)

    # Return the array of estimated output positions
    return out_pos



def predict_beam_uniformly_from_aoa(aoa):
    """
    Predict beams uniformly based on the angles of arrival (AOA).

    Parameters:
    - aoa (numpy.ndarray or array-like): Angles of arrival with shape (N, 1),
      where N is the number of datapoints.

    Returns:
    - numpy.ndarray: Ordered list of indices of the closest predictor points,
      representing the predicted beams. Shape (N, K), where K is the total number
      of beams.

    The function computes the distance of each datapoint to each predictor point
    based on the angles of arrival (AOA). It returns an ordered list of indices
    representing the predicted beams for each datapoint.

    Note:
    - The input aoa should have shape (N, 1), where N is the number of datapoints.
    - The function uses a uniformly distributed set of predictor points in the
      range [-pi, pi] to compute the distance.
    - The output is an array with shape (N, K), where K is the total number of beams.
    - The indices in the output array represent the closest predictor points for each datapoint.

    Examples:
    >>> aoa = np.array([[0.1], [1.5], [-2.0]])
    >>> predict_beam_uniformly_from_aoa(aoa)
    array([[12, 13, 11, ...,  6,  7,  5],
           [11, 12, 10, ...,  5,  6,  4],
           [ 5,  6,  4, ..., 15,  0, 14]])

    """
    beam_predictions = np.zeros_like(aoa)

    beam_ori = np.arange(N_BEAMS * N_ARR) / (N_BEAMS * N_ARR - 1) * 2*np.pi - np.pi

    angl_diff_to_each_beam = aoa.reshape((-1, 1)) - beam_ori

    beam_predictions = np.argsort(abs(angl_diff_to_each_beam), axis=1)

    return beam_predictions


def circular_distance(a, b, l=256, sign=False):
    """
    Compute the circular distance between two beam indices.

    Parameters:
    - a (int): First beam index.
    - b (int): Second beam index.
    - l (int, optional): Total number of beam indices in the circle (default is 256).
    - sign (bool, optional): If True, considers a as predicted and b as truth (default is False).

    Returns:
    - int: Circular distance between the two beam indices.

    The function computes the circular distance between two beam indices, a and b,
    in a circular way. It considers all numbers written in a circle with 'l' numbers,
    then computes the shortest distance between any two numbers.

    Examples:
    >>> circular_distance(0, 5, l=256)
    5
    >>> circular_distance(0, 255, l=256)
    1
    >>> circular_distance(0, 250, l=256)
    6
    >>> circular_distance(0, 127, l=256)
    127

    Note:
    - If 'sign' is True, a is considered as predicted, and b as truth.
    - The distance is returned as a positive value unless 'sign' is True.
    """
    while a < 0:
        a = l - abs(a)
    while b < 0:
        b = l - abs(b)
        
    a = a % l if a >= l else a
    b = b % l if b >= l else b
    
    dist = a - b

    if abs(dist) > l/2:
        dist = l - abs(dist)

    return dist if sign else abs(dist)


def compute_acc(all_beams, only_best_beam, top_k=[1, 3, 5]):
    
    """
    Compute top-k accuracy given predicted and ground truth labels.

    Parameters:
    - all_beams (numpy.ndarray): Predicted beams with shape (N_SAMPLES, N_BEAMS),
      representing either ground truth beams sorted by receive power or predicted beams
      sorted by algorithm's confidence.
    - only_best_beam (numpy.ndarray): Ground truth or predicted optimal beam index with shape (N_SAMPLES, 1).
    - top_k (list, optional): List of integers representing the top-k values for accuracy calculation (default is [1, 3, 5]).

    Returns:
    - numpy.ndarray: Top-k accuracy values.

    The function computes top-k accuracy given predicted and ground truth labels.
    It works bidirectionally, allowing it to handle cases where 'all_beams' can represent
    either the ground truth beams sorted by receive power or the predicted beams sorted by
    the algorithm's confidence of being the best.

    Examples:
    >>> all_beams = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
    >>> only_best_beam = np.array([[1], [2], [3]])
    >>> compute_acc(all_beams, only_best_beam, top_k=[1, 3])
    array([0.6667, 1.    ])

    Note:
    - 'all_beams' is assumed to have shape (N_SAMPLES, N_BEAMS), where N_SAMPLES is the number of samples,
      and N_BEAMS is the number of beams.
    - 'only_best_beam' is assumed to have shape (N_SAMPLES, 1), representing either the ground truth or predicted optimal beam index.
    - 'top_k' is a list of integers specifying the top-k values for accuracy calculation.
    """
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)

    n_test_samples = len(only_best_beam)
    if len(all_beams) != n_test_samples:
        raise Exception(
            'Number of predicted beams does not match number of labels.')

    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(only_best_beam)):
        for k_idx in range(n_top_k):
            hit = np.any(all_beams[samp_idx, :top_k[k_idx]] == only_best_beam[samp_idx])
            total_hits[k_idx] += 1 if hit else 0

    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(only_best_beam), 4)


def APL(true_best_pwr, est_best_pwr):
    """
    Calculate the Average Power Loss (APL).

    Parameters:
    - true_best_pwr (numpy.ndarray): True best power values.
    - est_best_pwr (numpy.ndarray): Estimated best power values.

    Returns:
    - float: Average Power Loss.

    The function computes the average of the power wasted by using the predicted beam
    instead of the ground truth optimum beam.

    Examples:
    >>> true_best_pwr = np.array([1.0, 2.0, 3.0])
    >>> est_best_pwr = np.array([1.5, 2.5, 3.5])
    >>> APL(true_best_pwr, est_best_pwr)
    0.1761

    Note:
    - 'true_best_pwr' and 'est_best_pwr' should have the same shape.
    - The Average Power Loss is calculated as the average of 10 * log10(est_best_pwr / true_best_pwr).
    """
    
    return np.mean(10 * np.log10(est_best_pwr / true_best_pwr))

