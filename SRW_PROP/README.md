# SRW_PROP
This project contains several tools to optimize the propagation parameters
of SRW simulator.

## Notes & Issues
1. PSO is still in the old version which require us to split the `varParam`.
Since its performance is not good, you wont use it in general.
   
2. some functions in the scripts can not be directly used in the optimization. You may
    have to verify the order of function argument such that `v` is always the first argument and
    others have default values.

3. `Coordinate Ascent` and `Coordinate Ascent with Momentum` only work under large learning rate (eg. 5.0).
For too small learning rates, they will stuck and do not update the propagation parameters.
   But in large learning rate, they work fine.

## Configuration
### Prerequisites
- SRW: `git clone https://github.com/ochubar/SRW.git`
- Pytorch == 1.4.0:

### Add `SRW` as Python Package
Before testing the propagation parameter optimization, you should 
add `SRW` as a package of your Python. This can be achieved by 
copying the `SRW.pth` into the site-packages folder of your Python
and change the path as `PATH_TO_SRW\env\work\srw_python` in `SRW.pth` to the path of the SRW package after
the git.

## Optimizers
### General Usage

In order to use the implemented optimizers in this package, we only need to download python
scripts from Sirepo, and import the following components:
    
- varParam [List]: the whole parameters (physics and propagation parameters). e.g.
    ```python
    setting_params = [
        ['name', 's', 'NSLS-II FMX beamline', 'simulation name'],

        #---Data Folder
        ['fdir', 's', '', 'folder (directory) name for reading-in input and saving output data files'],
    
        #---Electron Beam
        ['ebm_nm', 's', '', 'standard electron beam name'],
        ['ebm_nms', 's', '', 'standard electron beam name suffix: e.g. can be Day1, Final'],
        ['ebm_i', 'f', 0.5, 'electron beam current [A]'],
        ['ebm_e', 'f', 3.0, 'electron beam avarage energy [GeV]'],
        ['ebm_de', 'f', 0.0, 'electron beam average energy deviation [GeV]'],
        ['ebm_x', 'f', 0.0, 'electron beam initial average horizontal position [m]'],
        ['ebm_y', 'f', 0.0, 'electron beam initial average vertical position [m]'],
        ['ebm_xp', 'f', 0.0, 'electron beam initial average horizontal angle [rad]'],
        ['ebm_yp', 'f', 0.0, 'electron beam initial average vertical angle [rad]'],
        ['ebm_z', 'f', 0., 'electron beam initial average longitudinal position [m]'],
        ['ebm_dr', 'f', -0.7927500000000001, 'electron beam longitudinal drift [m] to be performed before a required calculation'],
        ['ebm_ens', 'f', 0.00089, 'electron beam relative energy spread'],
        ['ebm_emx', 'f', 5.500000000000001e-10, 'electron beam horizontal emittance [m]'],
        ['ebm_emy', 'f', 8e-12, 'electron beam vertical emittance [m]'],
        # Definition of the beam through Twiss:
        ['ebm_betax', 'f', 2.02, 'horizontal beta-function [m]'],
        ['ebm_betay', 'f', 1.06, 'vertical beta-function [m]'],
        ['ebm_alphax', 'f', 0.0, 'horizontal alpha-function [rad]'],
        ['ebm_alphay', 'f', 0.0, 'vertical alpha-function [rad]'],
        ['ebm_etax', 'f', 0.0, 'horizontal dispersion function [m]'],
        ['ebm_etay', 'f', 0.0, 'vertical dispersion function [m]'],
        ['ebm_etaxp', 'f', 0.0, 'horizontal dispersion function derivative [rad]'],
        ['ebm_etayp', 'f', 0.0, 'vertical dispersion function derivative [rad]'],
    ]
    ```

- set_up_funcs [List]: the list of set-up functions in the downloaded script used to run the experiments. e.g.
    ```python
    def set_optics(v=None):
        el = []
        pp = []
        for el_name in names:
            if el_name == 'S0':
                # S0: aperture 33.1798m
                el.append(srwlib.SRWLOptA(
                    _shape=v.op_S0_shape,
                    _ap_or_ob='a',
                    _Dx=v.op_S0_Dx,
                    _Dy=v.op_S0_Dy,
                    _x=v.op_S0_x,
                    _y=v.op_S0_y,
                ))
                pp.append(v.op_S0_pp)
            elif el_name == 'S0_HFM':
                # S0_HFM: drift 33.1798m
                el.append(srwlib.SRWLOptD(
                    _L=v.op_S0_HFM_L,
                ))
                pp.append(v.op_S0_HFM_pp)
            elif el_name == 'At_Sample':
                # At_Sample: watch 63.3m
                pass
        pp.append(v.op_fin_pp)
        return srwlib.SRWLOptC(el, pp)

    def setup_magnetic_measurement_files(v, filename="configurations/magn_meas_srx.zip"):
        import os
        import re
        import zipfile
        z = zipfile.ZipFile(filename)
        z.extractall()
        for f in z.namelist():
            if re.search(r'\.txt', f):
                v.und_mfs = os.path.basename(f)
                v.und_mdir = os.path.dirname(f) or './'
                return
        raise RuntimeError('missing magnetic measurement index *.txt file')
    
    # set-up functions.
    set_up_funcs = [setup_magnetic_measurement_files, set_optics]
    ```
    some functions in the scripts can not be directly used in the optimization. You may
    have to verify the order of function argument such that `v` is always the first argument and
    others have default values.

- tunable_params [Dict(tuple, tuple)]: the lists of tunable positions in the propagation parameters.
the keys denote the position in the `propagation_params` and the values are the valid range of the parameter.
    


### Implemented Methods
#### 1. Grid Search
The **Grid Search** method splits the parameter spaces into grid by fixed step-size.
Starting from the minimum parameter values given by `tunable_params`, it gradually 
increases the parameters through the grid. The class is initiated as
```python
OptimGridSearch(params, tunable_params, set_up_funcs, step_size)
```
where the parameters are given as

- params: the parameter list.
- tunable_params: the positions and ranges of tunable **propagation parameters**.
- set_up_funcs: the set-up function required (and also different) for each beamline.
- step_size (suggest value = 1.0): the step-size of each iteration.

A example of usage is given as 
```python
from optimizers.utils_optim import OptimGridSearch
from configurations.fmx_sample import *

if __name__ == "__main__":
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
    Optimizer = OptimGridSearch(varParam, tunable_params, [set_optics], step_size=0.1)
    Optimizer.forward(saveto='./results/test.json')
    print()
```

#### 2. Coordinate Ascent
The **coordinate Ascent** first compute a (pseudo) graident of a parameter and then upgrade it. 
The class is defined as 
```python
OptimGradientCoordinate(params, tunable_params, set_up_funcs, step_size=1.0, learning_rate=1e-2)
```
where the parameters are given as

- params: the parameter list.
- tunable_params: the positions and ranges of tunable **propagation parameters**.
- set_up_funcs: the set-up function required (and also different) for each beamline.
- step_size _(suggest value = 0.1)_: the step-size to approximate the gradient. 
- learning_rate _(suggest value = 5.0)_: the learning rate to update the parameters.

Usage is same as **Grid Search** except the class definition.

#### 3. Coordinate Ascent with Momentum
The **coordinate Ascent with Momentum** first compute a (pseudo) graident of a parameter 
and then upgrade it. Furthermore, it will save the previous update term as momentum
to speed up the convergence.
The class is defined as 
```python
OptimMomentumCoordinate(params, tunable_params, set_up_funcs, step_size=1.0, 
                        learning_rate=1e-2, beta=0.9)
```
where the parameters are given as

- params: the parameter list.
- tunable_params: the positions and ranges of tunable **propagation parameters**.
- set_up_funcs: the set-up function required (and also different) for each beamline.
- step_size _(suggest value = 0.1)_: the step-size to approximate the gradient. 
- learning_rate _(suggest value = 5.0)_: the learning rate to update the parameters.
- beta _(suggest value = 0.9)_: coefficent to control the contribution of momentum term.

Usage is same as **Grid Search** except the class definition.


#### 4. Particle Swarms Optimization

The **particle swarm optimization (PSO)** randomly initiates a population of
candidate solutions (particles). At each iteration, the particles are updated 
by the global best particle and the best historical particle of their own.
The interface is defined as 
```python
OptimParticleSwarm(params, tunable_params, set_up_funcs)
```
where the parameters are given as

- params: the parameter list.
- tunable_params: the positions and ranges of tunable **propagation parameters**.
- set_up_funcs: the set-up function required (and also different) for each beamline.

and

```python
OptimParticleSwarm(*args, **kargs).forward(num_particles, velocity_range, inertia_coeff, 
            cognitive_coeff,  social_coeff, step_size, num_steps, 
            early_stopping, saveto)
```
where the parameters are given as

- num_particles: number of particles.
- velocity_range: the maximum value of the velocity.
- inertia_coeff: inertia coefficient of PSO.
- cognitive_coeff: cognitive coefficient of PSO.
- social_coeff: social coefficient of PSO.
- step_size: the learning rate of each iteration.
- num_steps: the maximum number of iterations.
- early_stopping: the waiting iterations to terminate if no improvment.
- saveto: path to save the optimization results.

A example of usage is given as 
```python
from optimizers.utils_optim import OptimParticleSwarm
from configurations.fmx_sample_2 import *

if __name__ == "__main__":
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
    Optimizer = OptimParticleSwarm(names, setting_params, physics_params, propagation_params, tunable_params,
                                      set_up_funcs)
    Optimizer.forward(velocity_range=0.50, num_particles=5,inertia_coeff=0.5, cognitive_coeff=1.5,
                      social_coeff=1.5, step_size=0.5, saveto='TrueValues/ParticleSwarm/smi_sample.json')
```
