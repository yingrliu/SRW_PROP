import time
import numpy as np
from utils_optim import *
from configurations.fmx_sample_2_watchpoint1 import *

num_samples = 20
if __name__ == "__main__":
    # grid_search(names, setting_params, physics_params, propagation_params, index_list, 
    #             min_range=0.5, max_range=5.0, min_resolution=1, max_resolution=10, num_points=num_samples,
    #             img_path='cache')
    coordinate_ascent(names, setting_params, physics_params, propagation_params, index_list, 
                min_range=0.5, max_range=5.0, min_resolution=1, max_resolution=10, step_size=0.25,
                img_path='cache')
