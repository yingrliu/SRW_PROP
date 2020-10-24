# Author: Yingru Liu
# Institute: Stony Brook University
import sys
sys.path.append('..')
import torch
from optimizers.utils_optim import OptimReinforce, OptimDDPG
from configurations.fmx_sample import *

# Init.
BufferConfig = {
    "limit":1000, "window_length": 5, "ignore_episode_boundaries": False
}
OpticNetConfig = {
    "dimEmbed": 128, "dimNameRNN": 128, "dimProp": 4, "dimArray": 128, "dimCNN_channels": [32, 32, 32, 32, 32],
    "dimCNN_kernels": [3, 3, 3, 3, 3], "dimCNN_denses": [256, 256]
}
tunable_params = {}
for item in index_list:
    tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 7.5]
# test = Reinforce(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
#                  OpticNetConfig, device="cuda:0", checkpoint=None) # 'checkpoints/model_params.pth'
test = OptimDDPG(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
            BufferConfig, OpticNetConfig, device="cuda:0", checkpoint=None) # 'checkpoints/ddpg/model_params.pth'
# Compute Output.
test.train(5, ["configurations.chx_sample", "configurations.chx_sample",
               "configurations.csx_sample", "configurations.esm_sample", "configurations.hxn_sample",
               "configurations.smi_sample", "configurations.srx_sample"], saveto='checkpoints/ddpg')
test.forward(saveto='./test.json')