from optimizers.utils_optim import *
from configurations.smi_sample import *

if __name__ == "__main__":
    # todo:
    # index_list = input()~~
    coordinate_ascent(names, setting_params, physics_params, propagation_params, index_list, set_up_funcs,
                min_range=0.75, max_range=5.0, min_resolution=1, max_resolution=10, step_size=0.25,
                img_path='cache', saveto='results/srx.json')
