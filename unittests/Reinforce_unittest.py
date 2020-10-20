# Author: Yingru Liu
# Institute: Stony Brook University
# unit-test for the OpticNet Modules (optimizers.utils_optim.Reinforce).
import sys
sys.path.append('..')
import torch
from optimizers.utils_optim import Reinforce
from configurations.fmx_sample_2 import *


# Init.
OpticNetConfig = {
    "dimEmbed": 5, "dimNameRNN": 25, "dimProp": 4, "dimArray": 64, "dimCNN_channels": [32, 32, 32],
    "dimCNN_kernels": [3, 3, 3], "dimCNN_denses": [50, 50]
}
tunable_params = {}
for item in index_list:
    tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
test = Reinforce(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
                 OpticNetConfig, device="cuda:0")
test._initiation()
test._get_tensor_batch(5)
# Compute Output.
