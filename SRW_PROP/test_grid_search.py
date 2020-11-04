from optimizers.utils_optim import OptimCoordinateAscent, OptimParticleSwarm, OptimGeneticAlgorithm
from configurations.fmx_sample import *

if __name__ == "__main__":
    # todo:
    # index_list = input()~~
    tunable_params = {}
    for item in index_list:
        tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 10.]
    # Optimizer = OptimCoordinateAscent(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs,
    #                               step_size=0.10)
    Optimizer = OptimParticleSwarm(names, setting_params, physics_params, propagation_params, tunable_params,
                                      set_up_funcs)
    # Optimizer = OptimGeneticAlgorithm(names, setting_params, physics_params, propagation_params, tunable_params,
    #                                set_up_funcs, step_size=0.1)
    # Optimizer.forward(saveto=None)
    Optimizer.forward(velocity_range=0.50, inertia_coeff=0.5, cognitive_coeff=1.5,
                      social_coeff=1.5, step_size=0.5, saveto=None) # 'TrueValues/ParticleSwarm/smi_sample.json'
    # Optimizer.forward(num_particles=10, prob_crossover=0.3, prob_mutation=0.1)
    print()