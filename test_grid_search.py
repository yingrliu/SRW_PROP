import time
import numpy as np
import matplotlib.pyplot as plt
from utils_optim import *
from configurations.fmx_sample_2 import *

if __name__ == "__main__":
    # grid_search(names, setting_params, physics_params, propagation_params, index_list, 
    #             min_range=0.5, max_range=5.0, min_resolution=1, max_resolution=10, num_points=20,
    #             img_path='cache')
    coordinate_ascent(names, setting_params, physics_params, propagation_params, index_list, set_optics,
                min_range=0.5, max_range=5.0, min_resolution=1, max_resolution=10, step_size=0.25,
                img_path='cache')
    # coordinate_ascent_greedy(names, setting_params, physics_params, propagation_params, index_list,
    #                   min_range=0.5, max_range=5.0, min_resolution=1, max_resolution=10, step_size=0.25,
    #                   img_path='cache')
