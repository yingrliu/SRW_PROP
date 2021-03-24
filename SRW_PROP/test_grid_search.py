from optimizers.utils_optim import OptimGridSearch, OptimParticleSwarm, \
    OptimGradientCoordinate, OptimMomentumCoordinate
from configurations.srx_sample import *

if __name__ == "__main__":
    # todo:
    # index_list = input()~~
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [.75, 5.0] if item[-1] in [5, 7] else [.5, 10.]
    # Optimizer = OptimParticleSwarm(varParam, tunable_params, [set_optics])
    Optimizer = OptimMomentumCoordinate(varParam, tunable_params, [set_optics, setup_magnetic_measurement_files],
                                        step_size=1.0, learning_rate=5.0)   #, learning_rate=5.0
    Optimizer.forward(saveto=None)      #"results/GradientCoordinate/fmx_sample_0_p_0_0.json"
    # Optimizer.forward(num_particles=5, velocity_range=0.50, inertia_coeff=0.8, cognitive_coeff=1.5,
    #                   social_coeff=3.0, step_size=0.2, saveto=None)
    print()