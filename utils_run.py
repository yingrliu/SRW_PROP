# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to run the simulator, revised from Sirepo export samples.

import srwl_bl
import srwlib
import srwlpy
import srwl_uti_smp
import numpy as np

#############################################################################
#                       Experiment Running Tools                            #
#############################################################################
# set optics parameters. copy from Sirepo website.
def set_optics(names, v=None):
    el = []
    pp = []
    for el_name in names:
        if el_name == 'Aperture':
            # Aperture: aperture 20.0m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_Aperture_shape,
                _ap_or_ob='a',
                _Dx=v.op_Aperture_Dx,
                _Dy=v.op_Aperture_Dy,
                _x=v.op_Aperture_x,
                _y=v.op_Aperture_y,
            ))
            pp.append(v.op_Aperture_pp)
        elif el_name == 'Aperture_HFM':
            # Aperture_HFM: drift 20.0m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Aperture_HFM_L,
            ))
            pp.append(v.op_Aperture_HFM_pp)
        elif el_name == 'HFM':
            # HFM: sphericalMirror 42.0m
            el.append(srwlib.SRWLOptMirSph(
                _r=v.op_HFM_r,
                _size_tang=v.op_HFM_size_tang,
                _size_sag=v.op_HFM_size_sag,
                _nvx=v.op_HFM_nvx,
                _nvy=v.op_HFM_nvy,
                _nvz=v.op_HFM_nvz,
                _tvx=v.op_HFM_tvx,
                _tvy=v.op_HFM_tvy,
                _x=v.op_HFM_x,
                _y=v.op_HFM_y,
            ))
            pp.append(v.op_HFM_pp)

        elif el_name == 'HFM_SSA':
            # HFM_SSA: drift 42.0m
            el.append(srwlib.SRWLOptD(
                _L=v.op_HFM_SSA_L,
            ))
            pp.append(v.op_HFM_SSA_pp)
        elif el_name == 'SSA':
            # SSA: aperture 55.0m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_SSA_shape,
                _ap_or_ob='a',
                _Dx=v.op_SSA_Dx,
                _Dy=v.op_SSA_Dy,
                _x=v.op_SSA_x,
                _y=v.op_SSA_y,
            ))
            pp.append(v.op_SSA_pp)
        elif el_name == 'SSA_KB_Aperture':
            # SSA_KB_Aperture: drift 55.0m
            el.append(srwlib.SRWLOptD(
                _L=v.op_SSA_KB_Aperture_L,
            ))
            pp.append(v.op_SSA_KB_Aperture_pp)
        elif el_name == 'KB_Aperture':
            # KB_Aperture: aperture 66.0m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_KB_Aperture_shape,
                _ap_or_ob='a',
                _Dx=v.op_KB_Aperture_Dx,
                _Dy=v.op_KB_Aperture_Dy,
                _x=v.op_KB_Aperture_x,
                _y=v.op_KB_Aperture_y,
            ))
            pp.append(v.op_KB_Aperture_pp)
        elif el_name == 'KBv':
            # KBv: ellipsoidMirror 66.0m
            el.append(srwlib.SRWLOptMirEl(
                _p=v.op_KBv_p,
                _q=v.op_KBv_q,
                _ang_graz=v.op_KBv_ang,
                _size_tang=v.op_KBv_size_tang,
                _size_sag=v.op_KBv_size_sag,
                _nvx=v.op_KBv_nvx,
                _nvy=v.op_KBv_nvy,
                _nvz=v.op_KBv_nvz,
                _tvx=v.op_KBv_tvx,
                _tvy=v.op_KBv_tvy,
                _x=v.op_KBv_x,
                _y=v.op_KBv_y,
            ))
            pp.append(v.op_KBv_pp)

        elif el_name == 'KBv_KBh':
            # KBv_KBh: drift 66.0m
            el.append(srwlib.SRWLOptD(
                _L=v.op_KBv_KBh_L,
            ))
            pp.append(v.op_KBv_KBh_pp)
        elif el_name == 'KBh':
            # KBh: ellipsoidMirror 66.5m
            el.append(srwlib.SRWLOptMirEl(
                _p=v.op_KBh_p,
                _q=v.op_KBh_q,
                _ang_graz=v.op_KBh_ang,
                _size_tang=v.op_KBh_size_tang,
                _size_sag=v.op_KBh_size_sag,
                _nvx=v.op_KBh_nvx,
                _nvy=v.op_KBh_nvy,
                _nvz=v.op_KBh_nvz,
                _tvx=v.op_KBh_tvx,
                _tvy=v.op_KBh_tvy,
                _x=v.op_KBh_x,
                _y=v.op_KBh_y,
            ))
            pp.append(v.op_KBh_pp)

        elif el_name == 'KBh_Sample':
            # KBh_Sample: drift 66.5m
            el.append(srwlib.SRWLOptD(
                _L=v.op_KBh_Sample_L,
            ))
            pp.append(v.op_KBh_Sample_pp)
        elif el_name == 'Sample':
            # Sample: watch 67.0m
            pass
    pp.append(v.op_fin_pp)
    return srwlib.SRWLOptC(el, pp)

# run the experment based on the params. Modified from Sirepo website.
def run_experiment(names, setting_params, physics_params, propagation_params):
    """
    names - [list] the name list of instruments.
    setting_params - [list] the meta information of the experiment.
    physics_params - [list] .
    propagation_params - [list] .
    """
    varParam = srwl_bl.srwl_uti_ext_options(setting_params + physics_params + propagation_params)
    v = srwl_bl.srwl_uti_parse_options(varParam, use_sys_argv=True)
    op = set_optics(names, v)
    # this part is different for different beamline?
    # v.ss = True
    # v.ss_pl = 'e'
    # v.sm = True
    # v.sm_pl = 'e'
    # v.pw = True
    # v.pw_pl = 'xy'
    # v.si = True
    # v.si_pl = 'xy'
    # v.tr = True
    # v.tr_pl = 'xz'
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
    command_lines = [command_lines[-4], command_lines[-1]]
    dims = [0, 0]
    for i in range(2):
        term = command_lines[i].split(' ')[0]
        dims[-1 - i] = int(term[1:])
    result = np.reshape(result, newshape=dims)
    return result, dims