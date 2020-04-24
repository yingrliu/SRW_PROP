import time
import numpy as np
import matplotlib.pyplot as plt
from utils_optim import *
from configurations.srx_sample import *

# todo: try chx, hxn.
if __name__ == "__main__":
    coordinate_ascent(names, setting_params, physics_params, propagation_params, index_list, set_up_funcs,
                min_range=0.75, max_range=5.0, min_resolution=1, max_resolution=10, step_size=0.25,
                img_path='cache')
