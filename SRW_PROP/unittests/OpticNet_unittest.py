# Author: Yingru Liu
# Institute: Stony Brook University
# unit-test for the OpticNet Modules (optimizers.NN_package.Modules.OpticNet).
import torch
from optimizers.NN_package.Modules import OpticNet


# Init.
dimEmbed = 5
dimRNN = 25
dimProp = 4
dimArray = 64
dimCNN_channels = [32, 32, 32]
dimCNN_kernels = [3, 3, 3]
dimCNN_denses = [50, 50]
test = OpticNet(dimEmbed, dimRNN, dimProp, dimArray, dimCNN_channels, dimCNN_kernels, dimCNN_denses).cuda()
# Compute Output.
Names = ['Lens', 'Drift', 'Zone_Plate', 'Watchpoint']
Arrays = torch.randn(size=(1, 1, dimArray, dimArray)).cuda()
Prop = torch.ones(size=(len(Names), 5, 4)).cuda()
prop_new, delta_prop = test.forward(Names, Prop, Arrays)
assert prop_new.size() == (4, 5, 4)
assert delta_prop.size() == (4, 5, 4)
