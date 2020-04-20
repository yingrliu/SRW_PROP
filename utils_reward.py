# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to define the reward of operations.

import numpy as np
import scipy.signal as sci_signal
from scipy import interpolate
from skimage.metrics import structural_similarity as ssim

#############################################################################
#                         Currently Useless Tool                            #
#############################################################################
# access the system state from the propagation parameters.
def get_state(params):
    state = []
    for item in params:
        state.append([item[0], item[2][5:9]])
    return state

#############################################################################
#                        Reward Evaluation Tools                            #
#############################################################################
# compute the disimilarity between two 2-d arrays by 1 - SSIM.
def get_difference(array1, array2, prev_prop_params, prop_params):
    # interpolation array1, so its size is same as array2. 
    shape1 = array1.shape
    shape2 = array2.shape
    normal_term = max(array1.max(), array2.max()) + 1e-8
    array1, array2 = array1 / normal_term, array2 / normal_term
    # Compute the scale of the two arrays according to the range parameters.
    scale1, scale2 = [1.0, 1.0], [1.0, 1.0]
    for n in range(len(prop_params)):
        scale1[0] *= prev_prop_params[n][2][5]
        scale1[1] *= prev_prop_params[n][2][7]
        scale2[0] *= prop_params[n][2][5]
        scale2[1] *= prop_params[n][2][7]
    if shape1 != shape2:
        x1 = np.linspace(-scale1[1], scale1[1], shape1[1])
        y1 = np.linspace(-scale1[0], scale1[0], shape1[0])
        f = interpolate.interp2d(x1, y1, array1, kind='quintic')
        x2 = np.linspace(-scale2[1], scale2[1], shape2[1])
        y2 = np.linspace(-scale2[0], scale2[0], shape2[0])
        array1_int = f(x2, y2)
    else:
        array1_int = array1
    sim = ssim(array2, array1_int)
    return 1 - sim

# compute the penalty for runtime.
def get_complexity(prop_params, alpha=0.005, exponential=True):
    """
    :param: exponential: [bool] whether use exponential function to penalize more on large complexity.
    """
    complexity = 0
    for n, param in enumerate(prop_params):
        complexity += param[2][5] * param[2][6] * param[2][7] * param[2][8]
    complexity /= len(prop_params)
    return np.exp(alpha * complexity) if exponential else alpha * complexity



def get_image_ratio_reward(array, alpha=1e-2):
    shape = array.shape
    horizontal_range, vertical_range = 0, 0
    threshold = array.max() * alpha
    # find the horizontal range.
    for i in range(shape[0]):
        left = -1
        for j in range(shape[1]):
            if array[i, j] >= threshold:
                break
            left += 1
        right = shape[1]
        for j in range(shape[1] - 1, -1, -1):
            if array[i, j] >= threshold or left >= right:
                break
            right -= 1
        horizontal_range = max((right - left + 1) / shape[1], horizontal_range)
    # find the vertical range.
    for j in range(shape[1]):
        top = -1
        for i in range(shape[0]):
            if array[i, j] >= threshold:
                break
            top += 1
        buttom = shape[0]
        for i in range(shape[0] - 1, -1, -1):
            if array[i, j] >= threshold or top >= buttom:
                break
            buttom -= 1
        vertical_range = max((buttom - top + 1) / shape[0], vertical_range)
    if horizontal_range >= 1.0:
        horizontal_range = 0.1
    if vertical_range >= 1.0:
        vertical_range = 0.1
    return horizontal_range * vertical_range
    # if index[-1] == 5:
    #     return horizontal_range
    # else:
    #     return vertical_range


# todo: deprecated.
# # compute the ratio bewtween the central part and the whole part of a 2-d array.
# def get_image_ratio(array, index, edge_ratio=0.4):
#     """
#     edge_ratio: the ratio of edge with respect to the whole array shape.
#     index: the index of tuned parameters.
#     """
#     shape = array.shape
#     array = array.astype(np.float64)
#     array = (array - array.min()) / (array.max() -  array.min())
#     # print(array.min())
#     # print(array)
#     # array = (array > 0.01 * array.max()).astype(float)
#     edge_width = [int(shape[0] * edge_ratio) // 2, int(shape[1] * edge_ratio) // 2]
#     # if index[-1] == 5:
#     #     array_central_horizontal = array[:, edge_width[1]:shape[1] - edge_width[1]]
#     #     horizontal_ratio = np.mean((array_central_horizontal.sum(axis=1) + 1e-8) / (array.sum(axis=1) + 1e-8))
#     #     return horizontal_ratio
#     # else:
#     #     array_central_vertical = array[edge_width[0]:shape[0] - edge_width[0], :]
#     #     vertical_ratio = np.mean((array_central_vertical.sum(axis=0) + 1e-8) / (array.sum(axis=0) + 1e-8))
#     #     return vertical_ratio
#     array_central_horizontal = array[:, edge_width[1]:shape[1] - edge_width[1]]
#     horizontal_ratio = np.mean((array_central_horizontal.sum(axis=1) + 1e-8) / (array.sum(axis=1) + 1e-8))
#     array_central_vertical = array[edge_width[0]:shape[0] - edge_width[0], :]
#     vertical_ratio = np.mean((array_central_vertical.sum(axis=0) + 1e-8) / (array.sum(axis=0) + 1e-8))
#     return horizontal_ratio * vertical_ratio
# def get_image_ratio_reward(array, index, edge_ratio=0.2, lowerbound=0.98, upperbound=0.99):
#     image_ratio = get_image_ratio(array, index, edge_ratio)
#     lowerbound = max(lowerbound, 1. - edge_ratio)
#     # print(image_ratio)
#     if lowerbound <= image_ratio <= upperbound:
#         return 0.
#     elif image_ratio < lowerbound:
#         return np.exp(10 * np.abs(image_ratio - lowerbound))
#     else:
#         return np.exp(np.abs(image_ratio - upperbound))
# # smooth the data using a window with requested size.
# # copy from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
# def smooth(x, window_len=11, window='hanning'):
#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")
#     if x.size < window_len:
#         raise ValueError("Input vector needs to be bigger than window size.")
#     if window_len<3:
#         return x
#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
#     s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     #print(len(s))
#     if window == 'flat': #moving average
#         w=np.ones(window_len,'d')
#     else:
#         w=eval('np.'+window+'(window_len)')
#     y=np.convolve(w/w.sum(), s, mode='valid')
#     return y
#
# # compute the roughness as the distance of between the smoothed and original inputs.
# def roughness(array, Ln=1):
#     shape = array.shape[0]
#     # smoothed_array = smooth(array)
#     smoothed_array = sci_signal.savgol_filter(array, 5, 1)
#     if Ln == 1:
#         dis = np.abs(smoothed_array - array).mean()
#     elif Ln == 2:
#         dis = (smoothed_array - array) ** 2
#         dis = dis.mean()
#     else:
#         raise ValueError("The current version only support L1 and L2.")
#     return dis