from optimizers.utils_optim import OptimCoordinateAscent
from configurations.chx_sample import *

if __name__ == "__main__":
    # todo:
    # index_list = input()~~
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
    Optimizer = OptimCoordinateAscent(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
                                  step_size=0.1)
    Optimizer.forward(saveto='./TrueValues/chx_sample.json')
    print()
