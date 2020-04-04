# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to modulate the propagation parameters.

# Update the value of the range parameter positioned by the index.
def update_range(prop_params, index, value):
    if index[-1] not in (5, 7):
        raise ValueError("the range parameter index is given by [instrumentID, bit], and bit should be 5 or 7.")
    if index[0] >= len(prop_params):
        raise ValueError("The instrument ID exceeds the length of the beamline.")
    prop_params[index[0]][2][index[1]] = value
    return

# Batch operations to update the values of the propagation parameters.
def update_resolution(prop_params, index, value):
    if index[-1] not in (6, 8):
        raise ValueError("the resolution parameter index is given by [instrumentID, bit], and bit should be 6 or 8.")
    if index[0] >= len(prop_params):
        raise ValueError("The instrument ID exceeds the length of the beamline.")
    prop_params[index[0]][2][index[1]] = value
    return

def updata_param(prop_params, index, value):
    if index[-1] not in (5, 6, 7, 8):
        raise ValueError("the parameter index is given by [instrumentID, bit], and bit should be 5 ~ 8.")
    if index[0] >= len(prop_params):
        raise ValueError("The instrument ID exceeds the length of the beamline.")
    prop_params[index[0]][2][index[1]] = value
    return

def update_prop_params(prop_params, index_list, value_list):
    if len(index_list) != len(value_list):
        raise ValueError("The length of index_list is not equaled to the value_list")
    for n, index in enumerate(index_list):
        # if index[-1] == 6 or index[-1] == 8:
        #     update_resolution(prop_params, index, value_list[n])
        # elif index[-1] == 5 or index[-1] == 7:
        #     update_range(prop_params, index, value_list[n])
        updata_param(prop_params, index, value_list[n])

# save the propagation parameters as .json file.
def save_prop_params(prop_params, saveto=None):
    pass

