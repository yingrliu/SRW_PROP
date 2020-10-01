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
# compute the total reward of current step.
def get_total_rewards(array_new, array_old, mesh_new, mesh_old, prv_quality, params_new,
                      alpha_cpl=1e-2, alpha_ratio=1e-2, step_size=1.0):
    """

    :param array_new:
    :param array_old:
    :param mesh_new: [[horizontal_left, horizontal_right], [vertical_left, vertical_right]]
    :param mesh_old: [[horizontal_left, horizontal_right], [vertical_left, vertical_right]]
    :param prv_quality: previous image quality.
    :param params_new:
    :return:
    """
    complexity = get_complexity(prop_params=params_new, alpha=alpha_cpl)
    quality = get_difference(array_new, array_old, mesh_new, mesh_old, alpha_ratio=alpha_ratio) + prv_quality
    return quality / complexity, quality

#
def get_difference(array_new, array_old, mesh_new, mesh_old, alpha_ratio):
    # interpolation array1, so its size is same as array2. 
    shape_new = array_new.shape
    shape_old = array_old.shape
    normal_term = array_new.mean() + 1e-2
    array_new, array_old = array_new / normal_term, array_old / normal_term
    # Find the mesh of each array.
    meshx_new = np.linspace(mesh_new[0][0], mesh_new[0][1], shape_new[1])
    meshy_new = np.linspace(mesh_new[1][0], mesh_new[1][1], shape_new[0])
    meshx_old = np.linspace(mesh_old[0][0], mesh_old[0][1], shape_old[1])
    meshy_old = np.linspace(mesh_old[1][0], mesh_old[1][1], shape_old[0])
    # Find the mesh area of the new array that we are interested in.
    rangex, rangey = get_image_ratios(array_new, alpha_ratio)
    array_new, meshx_new, meshy_new = array_new[rangey[0]:rangey[1]+1, rangex[0]:rangex[1]+1], \
                                      meshx_new[rangex[0]:rangex[1]+1], meshy_new[rangey[0]:rangey[1]+1]
    # Interpolate the old array to the new mesh.
    f = interpolate.interp2d(meshx_old, meshy_old, array_old, kind='quintic')
    array_old = f(meshx_new, meshy_new)
    return np.sqrt(((array_new - array_old) ** 2).mean())

# compute the penalty for runtime.
def get_complexity(prop_params, alpha=0.01, exponential=True):
    """
    :param: exponential: [bool] whether use exponential function to penalize more on large complexity.
    """
    complexity = 1
    for n, param in enumerate(prop_params):
        complexity += param[2][5] * param[2][6] * param[2][7] * param[2][8]
    complexity /= len(prop_params)
    """
    # alpha =10. in this block.
    for idx in params_list:
        complexity *= prop_params[idx[0]][2][idx[1]]
    complexity /= len(params_list)
    """
    return np.exp(alpha * complexity) if exponential else alpha * complexity



def get_image_ratios(array, alpha=5e-2):
    shape = array.shape
    left, right, top, buttom = float('inf'), 0, float('inf'), 0
    threshold = array.max() * alpha
    # find the horizontal range.
    for i in range(shape[0]):
        temp = 0
        while temp < shape[1] and array[i, temp] < threshold:
            temp += 1
        left = min(left, temp)
        temp = shape[1] - 1
        while left < temp and array[i, temp] < threshold:
            temp -= 1
        if left < temp:
            right = max(right, temp)
    horizontal_range = [left, right]
    # find the vertical range.
    for j in range(shape[1]):
        temp = 0
        while temp < shape[0] and array[temp, j] < threshold:
            temp += 1
        top = min(top, temp)
        temp = shape[0] - 1
        while top < temp and array[temp, j] < threshold:
            temp -= 1
        if top < temp:
            buttom = max(buttom, temp)
    vertical_range = [top, buttom]
    return horizontal_range, vertical_range

