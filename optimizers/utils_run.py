# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to run the simulator, revised from Sirepo export samples.
import os, glob
import json
import srwl_bl
import srwlib
import srwlpy
import srwl_uti_smp
import numpy as np
import matplotlib.pyplot as plt


def run_experiment(names, setting_params, physics_params, propagation_params, set_up_funcs, plot=False):
    """
    names - [list] the name list of instruments.
    setting_params - [list] the meta information of the experiment.
    physics_params - [list] .
    propagation_params - [list] .
    """
    varParam = srwl_bl.srwl_uti_ext_options(setting_params + physics_params + propagation_params)
    v = srwl_bl.srwl_uti_parse_options(varParam, use_sys_argv=True)
    op = None
    for n, func in enumerate(set_up_funcs):
        value = func(v)
        if func.__name__ == 'set_optics':
            op = value
    if op is None:
        raise ValueError("set_optics() function should be included in set_up_funcs")
    # this part is different for different beamline?
    v.ws = True
    v.ws_pl = 'xy'
    mag = None
    if v.rs_type == 'm':
        mag = srwlib.SRWLMagFldC()
        mag.arXc.append(0)
        mag.arYc.append(0)
        mag.arMagFld.append(srwlib.SRWLMagFldM(v.mp_field, v.mp_order, v.mp_distribution, v.mp_len))
        mag.arZc.append(v.mp_zc)
    srwl_bl.SRWLBeamline(_name=v.name, _mag_approx=mag).calc_all(v, op)
    # remove saved image.
    if plot:
        plt.show()
    else:
        for file in glob.glob('*.png'):
            os.remove(file)
        plt.close('all')
    return


# read the .dat file.
def read_dat(file_path):
    result = np.loadtxt(file_path)
    f = open(file_path)
    lines = f.readlines()
    command_lines = []
    for line in lines:
        if line[0] == '#':
            command_lines.append(line)
        else:
            break
    f.close()
    point_lines = [command_lines[-4], command_lines[-1]]
    position_lines = command_lines[4:6] + command_lines[-3:-1]
    dims, horizontal_positions, vertical_positions = [0, 0], [0, 0], [0, 0]
    for i in range(2):
        term = point_lines[i].split(' ')[0]
        dims[-1 - i] = int(term[1:])
    for i in range(2):
        term = position_lines[i].split(' ')[0]
        horizontal_positions[i] = float(term[1:])
        term = position_lines[i+2].split(' ')[0]
        vertical_positions[i] = float(term[1:])
    result = np.reshape(result, newshape=dims)
    return result, horizontal_positions, vertical_positions

# save current params as .json file.
def save_params(physics_params, propagation_params, tuned_list, saveto):
    params = {}
    params['physics_params'] = physics_params
    params['propagation_params'] = propagation_params
    # TODO:
    # params['tuned_list'] = tuned_list
    with open(saveto, 'w') as outfile:
        json.dump(params, outfile)
    return


def load_params(json_file):
    params = json.load(json_file)
    physics_params = params['physics_params']
    propagation_params = params['propagation_params']
    return physics_params, propagation_params

# # mapping from number of parameters to instrument type.
# _type_dict={
#     ('_shape', '_Dx', '_Dy', '_x', '_y'): 'Aperture',
#     ('_L'): 'Drift',
#     ('_hfn', '_dim', '_r', '_size_tang', '_size_tag', '_ang', '_nvx', '_nvy', '_nvz', '_tvx', '_tvy', '_amp_coef', '_x',
#      '_y'): 'Circular_Cylinder',
#     ('_hfn', '_dim', '_p', '_q', '_ang', '_anp_coef', '_size_tang', '_size_tag', '_nvx', '_nvy', '_nvz', '_tvx', '_tvy',
#      '_x', '_y'): 'Elliptical_Cylinder',
#     (): 'Watchpoint',
#     10: 'CRL',
#     4: 'Lens',
#     6: 'Planar',
#     ('_size_tang', '_size_tag', '_nvx', '_nvy', '_nvz', '_tvx', '_tvy', '_x', '_y', '_m', '_grDen', '_grDen1',
#      '_grDen2', '_grDen3', '_grDen4'): 'Grating'
# }

# def extract_physics_params_type(physics_params, optic_list, ):
#     """
#     extract physics parameters for each instrument and figure its type. (e.g aperture...)
#     :param type_dict:
#     :param physics_params:
#     :return:
#     """
#     params_per_item = []
#     item_idx, item_params = len(optic_list) - 1, []
#     for content in physics_params[::-1]:
#         if item_idx >= 0 and optic_list[item_idx][0:-3] in content[0]:
#             item_params.insert(0, content)
#         elif item_idx < 0:
#             break
#         else:
#             params_per_item.insert(0, item_params)
#             item_idx -= 1
#             item_params = [content]
#     types = [_type_dict[len(param)] for param in params_per_item]
#     return