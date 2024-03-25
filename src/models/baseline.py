# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:17:57 2023

@author: Joao

This script reads a training CSV, outputs predictions for the data, and applies
the score metric to compute the final score. 

It also computes a few other known metrics, like top-k accuracy and average power loss.


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
    - lon_deltas (numpy.ndarray or array-like): Differences in longitudes.

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


# %% Read CSV and Load dataset
scen_idx = 36
csv_train = r'D:\Python\Multi Modal Beam Prediction with Deep Learning\data\raw\scenario36\deepsense_challenge2023_trainset.csv'
csv_dict_path = rf'D:\Python\Multi Modal Beam Prediction with Deep Learning\data\raw\scenario{scen_idx}\scenario{scen_idx}.p'

with open(csv_dict_path, 'rb') as fp:
    csv_dict = pickle.load(fp)

df_train = pd.read_csv(csv_train)

#%% Example of loading and displaying RGB180 images


if False: 
    import os
    sample_idx = 0

    # Extrair o diretório do arquivo CSV
    csv_train_folder = os.path.dirname(csv_train)

    # Extrair o diretório do cenário
    scen_folder = 'D:/Python/Multi Modal Beam Prediction with Deep Learning/data/raw/scenario' + str(df_train['scenario'][sample_idx])

    # Construir os caminhos completos para as imagens
    img1_path = os.path.join(csv_train_folder, scen_folder, csv_dict['unit1_rgb5'][sample_idx])
    img2_path = os.path.join(csv_train_folder, scen_folder, csv_dict['unit1_rgb6'][sample_idx])

    img1 = plt.imread(img1_path)
    img2 = plt.imread(img2_path)

    fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=200)
    axs[0].imshow(img1)  # frente
    axs[1].imshow(img2)  # trás
    plt.show()

# %% (Fast Loading) Load Training positions and ground truth positions

# Load all positions
samples_of_scen = np.where(df_train['scenario'] == scen_idx)[0]
n_samples = len(samples_of_scen)

loaded_positions = set()
train_positions = np.zeros((n_samples, X_SIZE, N_GPS, N_GPS_COORD))

y_pos1 = np.zeros((n_samples, N_GPS_COORD))
y_pos2 = np.zeros((n_samples, N_GPS_COORD))
y_pwrs = np.zeros((n_samples, N_ARR, N_BEAMS))


# Loop over each sample in the training data
for sample_idx in tqdm(range(n_samples), desc='Loading data'):
    # Get the current training sample
    train_sample = samples_of_scen[sample_idx]
    
    # Loop over each position index on the X axis
    for x_idx in range(X_SIZE):
        # Find the absolute relative index in the CSV DataFrame for the current training sample
        abs_idx_relative_index = (csv_dict[IDX_COL2] == df_train[f'x{x_idx+1}_'+IDX_COL1][train_sample])
        
        # Fill the training positions for the current sample and current X index
        # GPS positions of 'unit1' and 'unit2' are stored in train_positions
        # Index [0, :] is used for 'unit1' GPS coordinates, and index [1, :] is used for 'unit2' GPS coordinates
        train_positions[sample_idx, x_idx, 0, :] = csv_dict['unit1_gps1'][abs_idx_relative_index]
        train_positions[sample_idx, x_idx, 1, :] = csv_dict['unit2_gps1'][abs_idx_relative_index]

    # Positions of the output to compare with our position estimation approach
        # Find the index for the ground truth position 'y' in the CSV dictionary DataFrame
    y_idx = (csv_dict[IDX_COL2] == df_train['y1_'+IDX_COL1][train_sample])

    # Store the GPS positions for 'unit1' and 'unit2' corresponding to the ground truth 'y' position
    y_pos1[sample_idx] = csv_dict['unit1_gps1'][y_idx]
    y_pos2[sample_idx] = csv_dict['unit2_gps1'][y_idx]

    # Loop over each antenna array to store the power readings corresponding to the ground truth 'y' position
    for arr_idx in range(N_ARR):
        y_pwrs[sample_idx, arr_idx] = csv_dict[f'unit1_pwr{arr_idx+1}'][y_idx]


y_true_beams = df_train['y1_unit1_overall-beam'].values[samples_of_scen]

# array 1 (0-63), array 2 (64-127), array 3 (128-191), array 4 (192-255)
y_pwrs_reshaped = y_pwrs.reshape((n_samples, -1))
all_true_beams = np.flip(np.argsort(y_pwrs_reshaped, axis=1), axis=1)

# %% (Slow loading) Example of data loading for the testset

if False:
    import os

    # Set the test CSV file path same as train CSV file path
    csv_test = csv_train

    # Read the test CSV file into a DataFrame
    df_test = pd.read_csv(csv_test)

    # Number of samples in the test dataset
    n_samples = 1000  # Example size of the test set

    # Extract folder path from the test CSV file path
    folder = '/'.join(csv_test.split('/')[:-1])

    # Initialize an array to store GPS positions for the test dataset
    input_pos = np.zeros((n_samples, X_SIZE, N_GPS, N_GPS_COORD))

    # Loop over each sample in the test dataset
    for sample_idx in tqdm(range(n_samples), desc='Loading data'):
        # Loop over each position index on the X axis
        for x_idx in range(X_SIZE):
            # Construct the file path for 'unit1' GPS position
            gps_file_path = os.path.join('D:/Python/Multi Modal Beam Prediction with Deep Learning/data/raw/', 
                                         folder, 
                                         df_test[f'x{x_idx+1}_unit1_gps1'][sample_idx])
            try:
                # Load 'unit1' GPS position data into the input_pos array
                input_pos[sample_idx, x_idx, 0, :] = np.loadtxt(gps_file_path)
            except FileNotFoundError:
                # Handle FileNotFoundError if the GPS file is not found
                print(f"File not found: {gps_file_path}")

            # Construct the file path for 'unit2' GPS position
            gps_file_path = os.path.join('D:/Python/Multi Modal Beam Prediction with Deep Learning/data/raw/', 
                                         folder, 
                                         df_test[f'x{x_idx+1}_unit2_gps1'][sample_idx])
            try:
                # Load 'unit2' GPS position data into the input_pos array
                input_pos[sample_idx, x_idx, 1, :] = np.loadtxt(gps_file_path)
            except FileNotFoundError:
                # Handle FileNotFoundError if the GPS file is not found
                print(f"File not found: {gps_file_path}")

# %% Step 1: Estimate positions in the new timestamp (linear interpolation)
delta_input = 0.2  # time difference between input samples [s]
delta_output = 0.5  # time difference from last input to output [s]

gps1_est_pos = estimate_positions(train_positions[:, :, 0, :], delta_input, delta_output)
gps2_est_pos = estimate_positions(train_positions[:, :, 1, :], delta_input, delta_output)

# Compare estimated with real
if True:
    plt.figure(figsize=(10, 6), dpi=200)
    n = np.arange(10500)
    plt.plot(y_pos1[n, 0], -y_pos1[n, 1], alpha=.5,
             label='True positions', marker='o', markersize=1)
    plt.plot(gps1_est_pos[n, 0], -gps1_est_pos[n, 1], alpha=.5, 
             label='Estimated positions', c='r', marker='o', markersize=1)
    plt.annotate('start', xy=(y_pos1[0,0], -y_pos1[0,1]))
    plt.annotate('end', xy=(y_pos1[n[-1],0], -y_pos1[n[-1],1]))
    plt.title('Position Estimation')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.show()
    
# %% Step 2: With the estimated positions, estimate orientation

# 2.1 - Determine heading of vehicles using last available location and the new estimated location
# Calculate the difference in latitude and longitude between the estimated GPS positions and the last available GPS positions
lat_deltas = gps1_est_pos[:, 0] - train_positions[:, -1, 0, 0]
lon_deltas = gps1_est_pos[:, 1] - train_positions[:, -1, 0, 1]

# Compute the orientation (heading) based on the position deltas
heading = compute_ori_from_pos_delta(lat_deltas, lon_deltas)

# 2.2 - Determine relative position (converted to orientation) between vehicles
# Calculate the difference in latitude and longitude between the estimated GPS positions of vehicle 1 and vehicle 2
lat_deltas = gps1_est_pos[:, 0] - gps2_est_pos[:, 0]
lon_deltas = gps1_est_pos[:, 1] - gps2_est_pos[:, 1]

# Compute the orientation (relative position) based on the position deltas
ori_rel = compute_ori_from_pos_delta(lat_deltas, lon_deltas)

# Compute the estimated Angle of Arrival (AOA) using the relative orientation, heading, and a reference angle
# The relative orientation is subtracted from the heading and a constant angle (pi/4) is subtracted from the result
aoa_estimation = norm_2pi(-1 * (ori_rel - heading - np.pi / 4))

# Check if the estimated AOA correlates enough for an accurate prediction

if True:
    beams_angle = y_true_beams/255*2*np.pi - np.pi
    x = np.arange(len(aoa_estimation))
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(x, aoa_estimation, alpha=.5, s=3, zorder=2, label='aoa est')
    plt.scatter(x, beams_angle, alpha=.5, s=3, label='beams angle')
    plt.title('Position Estimation')
    plt.xlabel('Sample index')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.show()
    
# %% Step 3: From orientation, estimate beam (assume uniform beam distribution)

beam_pred_all = predict_beam_uniformly_from_aoa(aoa_estimation)
best_beam_pred = np.copy(beam_pred_all[:, 0]) # keep decoupled

# After analysis, we were often 1 beam short. 
pred_diff = np.array([circular_distance(a, b, sign=True)
                      for a, b in zip(best_beam_pred, y_true_beams)])

# The box sometimes is slightly rotated around it's Z axis, so we can shift our
# Beam predictions a constant offset to get better performance. Admit offsets up to 2.
# Note: this adjustment is only for the training phase
shift = -round(np.mean(pred_diff[abs(pred_diff) < 5]))
print(f'estimated_shift = {shift}')
beam_pred_all += shift
best_beam_pred += shift

# make sure the output is within 0-255
beam_pred_all[beam_pred_all>255] -= 255
beam_pred_all[beam_pred_all<0] += 255
best_beam_pred[best_beam_pred>255] -= 255
best_beam_pred[best_beam_pred<0] += 255

# Check if the prediction is good
if True:
    x = np.arange(len(aoa_estimation))
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(x, best_beam_pred, alpha=.5, s=3, zorder=2, label='pred')
    plt.scatter(x, y_true_beams, alpha=.5, s=3, label='true')
    plt.legend()
    plt.show()

#%% Output results for competition

df_out = pd.DataFrame()
df_out['prediction'] = best_beam_pred
df_out.to_csv('SUBMISSION-EXAMPLE_prediction.csv', index=False)
# put group name instead of "SUBMISSION-EXAMPLE"

# %% Compute Scores

# Note: only best_beam_pred and all_true_beams needed -> 
# Our evaluation script will have access to the ground truth beams of the testset
# and you submit the best_beam_pred in a csv

pred_diff_abs = np.array([circular_distance(a, b)
                          for a, b in zip(best_beam_pred, all_true_beams[:,0])])
average_beam_index_diff = np.mean(pred_diff_abs) # lower is better!

print(f'Average Beam Index Distance = {average_beam_index_diff:.2f}')

# Visualize beam diff
if True:
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(np.arange(len(pred_diff_abs)), pred_diff_abs, s=2)
    plt.hlines(y=average_beam_index_diff, xmin=0, xmax=len(pred_diff_abs), color='r',alpha=.5)

# "Probability of the prediction of the best beam being in the set of best k ground truth beams"
top_k = compute_acc(all_true_beams, best_beam_pred, top_k=[1, 3, 5])
print(f'Top-k = {top_k}')

# "Probability of the ground truth best beam being in the set of most likely k predicted beams"
top_k = compute_acc(beam_pred_all, all_true_beams[:, 0], top_k=[1, 3, 5])
print(f'(not used) Top-k = {top_k}')

# For practical submissions, we implement only the first way so we 
# only require the best predicted beam. 

# And finally, the score -> APL (Average Power Loss)
est_best_pwr = y_pwrs_reshaped[np.arange(n_samples), best_beam_pred]
true_best_pwr = y_pwrs_reshaped[np.arange(n_samples), all_true_beams[:, 0]]
apl = APL(true_best_pwr, est_best_pwr)
print(f'Average Power Loss = {apl:.2f} dB')