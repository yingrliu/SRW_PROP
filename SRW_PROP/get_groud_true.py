from optimizers.utils_run import run_experiment
from configurations.esm_sample import *

if __name__ == "__main__":
    run_experiment(names, setting_params, physics_params, propagation_params, set_up_funcs, plot=True)

