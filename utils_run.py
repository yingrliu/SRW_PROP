# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to run the simulator, revised from Sirepo export samples.
import os, glob
import srwl_bl
import srwlib
import srwlpy
import srwl_uti_smp
import numpy as np
import matplotlib.pyplot as plt


def run_experiment(names, setting_params, physics_params, propagation_params, set_up_funcs):
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