# Author: Yingru Liu
# Institute: Stony Brook University
import sys
sys.path.append('..')
import torch
from optimizers.utils_optim import Reinforce, DDPG
from configurations.fmx_sample import *

# Init.
BufferConfig = {
    "limit":1000, "window_length": 5, "ignore_episode_boundaries": False
}
OpticNetConfig = {
    "dimEmbed": 5, "dimNameRNN": 25, "dimProp": 4, "dimArray": 64, "dimCNN_channels": [32, 32, 32],
    "dimCNN_kernels": [3, 3, 3], "dimCNN_denses": [50, 50]
}
tunable_params = {}
for item in index_list:
    tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 7.5]
test = Reinforce(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
                 OpticNetConfig, device="cuda:0")
# test = DDPG(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
#             BufferConfig, OpticNetConfig, device="cuda:0")
# Compute Output.
# test.train(25, ["configurations.fmx_sample"], saveto='checkpoints')
test.forward(saveto='./results/test.json')

# , "configurations.chx_sample",
#                   "configurations.csx_sample", "configurations.esm_sample", "configurations.hxn_sample",
#                   "configurations.smi_sample", "configurations.srx_sample"