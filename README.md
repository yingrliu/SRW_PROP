# SRW_PROP
This project contains several tools to optimize the propagation parameters
of SRW simulator.


## Configuration
### Prerequisites
- SRW: `git clone https://github.com/ochubar/SRW.git`
- Skimage:

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
Coordinate_Ascent(names, setting_params, physics_params, prop_params, 
                  tunable_params, set_up_funcs)
```
where the parameters are given as

- names: the sequence of optics instrument names.
- setting_params: some global setting parameters.
- physics_params: the physics parameters of optics instruments.
- prop_params: the propagation parameters of optics instruments.
- tunable_params: the positions and ranges of tunable **propagation parameters**.
- set_up_funcs: the set-up function required (and also different) for each beamline.

A example of usage is given as 
```python
from optimizers.utils_optim import Coordinate_Ascent
from configurations.fmx_sample_2 import *

if __name__ == "__main__":
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
    Optimizer = Coordinate_Ascent(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
                                  step_size=0.1)
    Optimizer.forward(saveto='./results/test.json')
    print()
```
#### 1.2. Update Notes
[`Notes/Cordinate_Ascent.md`](Notes/Cordinate_Ascent.md)


### 2. REINFROCE
