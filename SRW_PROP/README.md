# SRW_PROP
This project contains several tools to optimize the propagation parameters
of SRW simulator.


## Configuration
### Prerequisites
- SRW: `git clone https://github.com/ochubar/SRW.git`
- Skimage:
- Pytorch == 1.4.0:

### Add `SRW` as Python Package
Before testing the propagation parameter optimization, you should 
add `SRW` as a package of your Python. This can be achieved by 
copying the `SRW.pth` into the site-packages folder of your Python
and change the path as `PATH_TO_SRW\env\work\srw_python` in `SRW.pth` to the path of the SRW package after
the git.

## Optimizers
### 1. Coordinate Ascent
#### 1.1. Descriptions
The **coordinate Ascent** is the simplest global search method to discover the optimal
propagation parameters by gradually increasing the values in the tunable parameter
list with a fix step-size. The interface is defined as 
```python
OptimCoordinateAscent(names, setting_params, physics_params, prop_params, 
                  tunable_params, set_up_funcs)
```
where the parameters are given as

- names: the sequence of optics instrument names.
- setting_params: some global setting parameters.
- physics_params: the physics parameters of optics instruments.
- prop_params: the propagation parameters of optics instruments.
- tunable_params: the positions and ranges of tunable **propagation parameters**.
- set_up_funcs: the set-up function required (and also different) for each beamline.
- step_size: the learning rate of each iteration.

A example of usage is given as 
```python
from optimizers.utils_optim import OptimCoordinateAscent
from configurations.fmx_sample_2 import *

if __name__ == "__main__":
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
    Optimizer = OptimCoordinateAscent(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
                                  step_size=0.1)
    Optimizer.forward(saveto='./results/test.json')
    print()
```

#### 1.2. Update Notes
[`Notes/Cordinate_Ascent.md`](Notes/Cordinate_Ascent.md)

### 2. Particle Swarms Optimization

#### 2.1. Descriptions
The **particle swarm optimization (PSO)** randomly initiates a population of
candidate solutions (particles). At each iteration, the particles are updated 
by the global best particle and the best historical particle of their own.
The interface is defined as 
```python
OptimParticleSwarm(names, setting_params, physics_params, prop_params, 
                  tunable_params, set_up_funcs)
```
where the parameters are given as

- names: the sequence of optics instrument names.
- setting_params: some global setting parameters.
- physics_params: the physics parameters of optics instruments.
- prop_params: the propagation parameters of optics instruments.
- tunable_params: the positions and ranges of tunable **propagation parameters**.
- set_up_funcs: the set-up function required (and also different) for each beamline.

and

```python
OptimParticleSwarm(*args, **kargs).forward(velocity_range, inertia_coeff, 
            cognitive_coeff,  social_coeff, step_size, num_steps, 
            early_stopping, saveto)
```
where the parameters are given as

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
    Optimizer.forward(velocity_range=0.50, inertia_coeff=0.5, cognitive_coeff=1.5,
                      social_coeff=1.5, step_size=0.5, saveto='TrueValues/ParticleSwarm/smi_sample.json')
    print()
```

#### 2.2. Update Notes
[`Notes/ParticleSwarm.md`](Notes/ParticleSwarm.md)

### 3. REINFROCE
#### 3.1. Descriptions

#### 3.2. Update Notes
[`Notes/REINFORCE.md`](Notes/REINFORCE.md)