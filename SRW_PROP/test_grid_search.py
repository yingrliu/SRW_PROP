from optimizers.utils_optim import OptimGridSearch, OptimParticleSwarm, \
    OptimGradientCoordinate, OptimMomentumCoordinate
from configurations.srx_sample import *

if __name__ == "__main__":
    # todo:
    # index_list = input()~~
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
    # Optimizer = OptimParticleSwarm(names, setting_params, physics_params, propagation_params, tunable_params,
    #                                   set_up_funcs)
    Optimizer = OptimGridSearch(names, setting_params, physics_params, propagation_params, tunable_params,
                                   set_up_funcs, step_size=1.0)                     #, learning_rate=1.0
    Optimizer.forward(saveto='results/GridSearch/srx_sample.json')
    # Optimizer.forward(num_particles=5, velocity_range=0.50, inertia_coeff=0.8, cognitive_coeff=1.5,
    #                   social_coeff=3.0, step_size=0.2, saveto='TrueValues/ParticleSwarm/srx_sample.json')
    print()