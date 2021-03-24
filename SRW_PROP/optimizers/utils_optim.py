# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to optimize the propagation parameters.
import copy
import glob
from optimizers.utils_reward import *
from optimizers.utils_run import *
from optimizers.utils_operation import *


class _optimizer():
    """
    Hyper class of all the optimizer.
    """
    def __init__(self, params, tunable_params, set_up_funcs, *args, **kwargs):
        """

        :param params: the  parameter list of the experiments.
        :param tunable_params: Dict[tuple1, tuple2] -- a dictionary indicating the position (tuple1) and range (tuple2) of the tunable prop params.
        :param set_up_funcs: set_up function for the beamline.
        """
        # TODO: add assert.
        self.other_params, self.prop_params, self.cache_path = split_params(params)
        self.tunable_params_positions, self.tunable_params_ranges = [], []
        for position, value in tunable_params.items():
            # check whether the index list is valid.
            if position[-1] < 5 or position[-1] > 8:  # range.
                raise ValueError("Invalid parameter index!")
            self.tunable_params_positions.append(position)
            self.tunable_params_ranges.append(value)
        self.set_prop_params()
        # save the parameters to run the experiments.
        self.set_up_func = set_up_funcs
        if self.cache_path and not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.result_files = glob.glob(os.path.join(self.cache_path, "res_int_*.dat"))
        self.result_files.remove(os.path.join(self.cache_path, "res_int_se.dat"))
        self.source_files = [os.path.join(self.cache_path, "res_int_se.dat")]
        self.arrays, self.prv_arrays, self.prv_prop_params, self.mesh, self.prv_mesh = None, None, None, None, None
        self.prv_quality = 0.
        return

    def forward(self, *args, **kwargs):
        return

    def set_prop_params(self, param_values=None):
        """
        set the propagation parameters given a dictionary.
        :param param_values: Dict[tuple1, float] -- a dictionary indicating the position (tuple1) and value (float) of the tunable prop params.
        :return:
        """
        if param_values is not None:
            for position, value in param_values.items():
                updata_param(self.prop_params, position, value)
        else:
            # initiate the tunable parameters to its smallest value.
            for position, value in zip(self.tunable_params_positions, self.tunable_params_ranges):
                updata_param(self.prop_params, position, value[0])
        return

    def _operate_experiments(self, plot=False):
        """
        run the SRW simulation.
        :param plot: whether visualize the results.
        :return:
        """
        run_experiment(self.other_params, self.prop_params, self.set_up_func, plot=plot)
        if not plot:
            plt.close()
        # compute the retio, reward and complexity.
        dat_records = [read_dat(path) for path in self.result_files]
        self.arrays = [record[0] for record in dat_records]
        self.mesh = [record[1:] for record in dat_records]
        return

    def _get_reward(self):
        """
        get the reward of the current experiment run.
        :return: rewards = [reward-score, accumulated quality, difference in this step, complexity]
        """
        param_list = []
        for position in self.tunable_params_positions:
            param_list.append(self.prop_params[position[0]][2][position[1]])
        rewards = np.asarray([get_total_rewards(array_new=current, array_old=prev, mesh_new=mesh, mesh_old=mesh_old,
                                                prv_quality=self.prv_quality, params_new=param_list)
                              for mesh_old, mesh, prev, current in zip(self.prv_mesh, self.mesh, self.prv_arrays, self.arrays)])
        return rewards

    def _initiation(self, init_values=None):
        """
        Initiate the optimization process.
        :param init_values: initial values of the tunable parameters. If None, use the minimum values instead.
        :return:
        """
        self.set_prop_params(init_values)
        self._operate_experiments(plot=False)
        self.prv_mesh = copy.deepcopy(self.mesh)
        self.prv_arrays, self.prv_quality = [np.zeros_like(array) for array in self.arrays], 0.
        self.prv_arrays, self.prv_quality = copy.deepcopy(self.arrays), self._get_reward()[:, 1].mean()
        return

    @staticmethod
    def _clean_cache(cache_path):
        # todo:
        return


class OptimGridSearch(_optimizer):
    """
    GridSearch for each dimension iteratively.
    """
    def __init__(self, params, tunable_params, set_up_funcs, step_size=0.20):
        """

        :param params: the  parameter list of the experiments.
        :param tunable_params: Dict[tuple1, tuple2] -- a dictionary indicating the position (tuple1) and range (tuple2) of the tunable prop params.
        :param set_up_funcs: set_up function for the beamline.
        :param step_size: the step_size of each iteration.
        """
        super(OptimGridSearch, self).__init__(params, tunable_params, set_up_funcs)
        self.step_size = step_size
        return

    def forward(self, saveto=None):
        """
        run the optimization process.
        :param saveto: a path to save the tuning results in .json.
        :return:
        """
        tuned_params = {}
        # Initiate the optimization process.
        self._initiation()
        total_updates, global_best_reward = 0, self._get_reward()[:, 0].mean()
        # Optimization process.
        print("------------> Tuning parameters.")
        updates = len(self.tunable_params_positions)
        runtimes = 0
        while updates:
            new_updates = 0
            for n, position in enumerate(self.tunable_params_positions):
                # define new values after update.
                prev_value = self.prop_params[position[0]][2][position[1]]
                new_value = self.prop_params[position[0]][2][position[1]] + self.step_size
                if not (self.tunable_params_ranges[n][0] < new_value < self.tunable_params_ranges[n][1]):
                    continue
                # update the mesh, mesh_old, array, array_old and get rewards.
                updata_param(self.prop_params, position, new_value)
                self._operate_experiments(plot=False)
                runtimes += 1
                rewards = self._get_reward()
                current_reward = rewards[:, 0].mean()
                print(position, '\t', new_value, '\t', current_reward)
                # check whether there is improvement.
                if current_reward - global_best_reward > 1e-4:
                    # update parameters.
                    new_updates += 1
                    tuned_params[position] = new_value
                    global_best_reward = current_reward
                    self.prv_arrays, self.prv_mesh, self.prv_quality = copy.deepcopy(self.arrays), \
                                                                       copy.deepcopy(self.mesh), rewards[:, 1].mean()
                else:
                    # reset the parameter.
                    updata_param(self.prop_params, position, prev_value)
            updates = new_updates
            total_updates += new_updates
        # Print the tunning results.
        print("------------> Finish Optimization.")
        print(tuned_params)
        print("\x1b[1;35mThe number of updates: {}; The number of exp runs: {}; the best reward: {}.\x1b[0m".format(
            total_updates, runtimes, global_best_reward))
        if saveto:
            save_params(self.other_params, self.prop_params, tuned_params, saveto)
        self._operate_experiments(plot=True)
        return tuned_params


class OptimGradientCoordinate(_optimizer):
    """
    Coordinate Ascent with the approximation of the gradient.
    """
    def __init__(self, params, tunable_params, set_up_funcs, step_size=1.0, learning_rate=1e-2):
        """

        :param params: the  parameter list of the experiments.
        :param tunable_params: Dict[tuple1, tuple2] -- a dictionary indicating the position (tuple1) and range (tuple2) of the tunable prop params.
        :param set_up_funcs: set_up function for the beamline.
        :param step_size: the step_size used to approximate the gradient.
        :param learning_rate: the learning rate (step-size) to update the parameters.
        """
        super(OptimGradientCoordinate, self).__init__(params, tunable_params, set_up_funcs)
        self.step_size = step_size
        self.learning_rate = learning_rate
        return

    def forward(self, saveto=None):
        tuned_params = {}
        # Initiate the optimization process.
        self._initiation()
        total_updates, global_best_reward = 0, self._get_reward()[:, 0].mean()
        # Optimization process.
        print("------------> Tuning parameters.")
        updates = len(self.tunable_params_positions)
        runtimes = 0
        while updates:
            new_updates = 0
            iterative_best_reward, tuned_position, tuned_value = 0., None, None
            for n, position in enumerate(self.tunable_params_positions):
                # approximate the gradient.
                prev_value = self.prop_params[position[0]][2][position[1]]
                new_value = self.prop_params[position[0]][2][position[1]] + self.step_size
                if not (self.tunable_params_ranges[n][0] < new_value < self.tunable_params_ranges[n][1]):
                    continue
                #
                updata_param(self.prop_params, position, new_value)
                self._operate_experiments(plot=False)
                runtimes += 1
                rewards = self._get_reward()
                gradient = rewards[:, 2].sum() / self.step_size
                # update the parameters.
                # new_value = prev_value + max(1, self.learning_rate * gradient // self.step_size) * self.step_size
                velocity = self.learning_rate * gradient
                if velocity < self.step_size:
                    continue
                new_value = prev_value + velocity
                updata_param(self.prop_params, position, new_value)
                self._operate_experiments(plot=False)
                runtimes += 1
                rewards = self._get_reward()
                current_reward = rewards[:, 0].mean()
                print(position, '\t', new_value, '\t', current_reward)
                # check whether there is improvement.
                if current_reward - global_best_reward > 5e-4:
                    # update parameters.
                    new_updates += 1
                    tuned_params[position] = new_value
                    global_best_reward = current_reward
                    self.prv_arrays, self.prv_mesh, self.prv_quality = copy.deepcopy(self.arrays), \
                                                                       copy.deepcopy(self.mesh), rewards[:, 1].mean()
                else:
                    # reset the parameter.
                    updata_param(self.prop_params, position, prev_value)
            updates = new_updates
            total_updates += new_updates
        # Print the tunning results.
        print("------------> Finish Optimization.")
        print(tuned_params)
        print("\x1b[1;35mThe number of updates: {}; The number of exp runs: {}; the best reward: {}.\x1b[0m".format(
            total_updates, runtimes, global_best_reward))
        if saveto:
            save_params(self.other_params, self.prop_params, tuned_params, saveto)
        self._operate_experiments(plot=True)
        return tuned_params


class OptimMomentumCoordinate(_optimizer):
    """
    Coordinate Ascent with the approximation of the gradient and momentum.
    """
    def __init__(self, params, tunable_params, set_up_funcs, step_size=0.10, learning_rate=1e-2, beta=0.9):
        """

        :param params: the  parameter list of the experiments.
        :param tunable_params: Dict[tuple1, tuple2] -- a dictionary indicating the position (tuple1) and range (tuple2) of the tunable prop params.
        :param set_up_funcs: set_up function for the beamline.
        :param step_size: the step_size used to approximate the gradient.
        :param learning_rate: the learning rate (step-size) to update the parameters.
        :param beta: coefficient of the momentum.
        """
        super(OptimMomentumCoordinate, self).__init__(params, tunable_params, set_up_funcs)
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.beta = beta
        return

    def forward(self, saveto=None):
        tuned_params = {}
        # Initiate the optimization process.
        self._initiation()
        total_updates, global_best_reward = 0, self._get_reward()[:, 0].mean()
        momentums = [0.] * len(self.tunable_params_positions)
        # Optimization process.
        print("------------> Tuning parameters.")
        updates = len(self.tunable_params_positions)
        runtimes = 0
        while updates:
            new_updates = 0
            iterative_best_reward, tuned_position, tuned_value = 0., None, None
            for n, position in enumerate(self.tunable_params_positions):
                # approximate the gradient.
                prev_value = self.prop_params[position[0]][2][position[1]]
                new_value = self.prop_params[position[0]][2][position[1]] + self.step_size
                if not (self.tunable_params_ranges[n][0] < new_value < self.tunable_params_ranges[n][1]):
                    continue
                #
                updata_param(self.prop_params, position, new_value)
                self._operate_experiments(plot=False)
                runtimes += 1
                rewards = self._get_reward()
                gradient = rewards[:, 2].sum() / self.step_size
                if momentums[n]:
                    velocity = self.learning_rate * gradient * (1. - self.beta) + momentums[n] * self.beta
                else:
                    velocity = self.learning_rate * gradient
                if velocity < self.step_size:
                    continue
                # update the parameters.
                new_value = prev_value + velocity
                updata_param(self.prop_params, position, new_value)
                self._operate_experiments(plot=False)
                runtimes += 1
                rewards = self._get_reward()
                current_reward = rewards[:, 0].mean()
                print(position, '\t', new_value, '\t', current_reward)
                # check whether there is improvement.
                if current_reward - global_best_reward > 5e-4:
                    # update parameters.
                    new_updates += 1
                    tuned_params[position] = new_value
                    global_best_reward = current_reward
                    self.prv_arrays, self.prv_mesh, self.prv_quality = copy.deepcopy(self.arrays), \
                                                                       copy.deepcopy(self.mesh), rewards[:, 1].mean()
                    momentums[n] = velocity
                else:
                    # reset the parameter.
                    updata_param(self.prop_params, position, prev_value)
            updates = new_updates
            total_updates += new_updates
        # Print the tunning results.
        print("------------> Finish Optimization.")
        print(tuned_params)
        print("\x1b[1;35mThe number of updates: {}; The number of exp runs: {}; the best reward: {}.\x1b[0m".format(
            total_updates, runtimes, global_best_reward))
        if saveto:
            save_params(self.other_params, self.prop_params, tuned_params, saveto)
        self._operate_experiments(plot=True)
        return tuned_params


class OptimParticleSwarm(_optimizer):
    """
    Determinisic Particle Swarm Optimization.
    """
    def __init__(self, params, tunable_params, set_up_funcs):
        """

        :param params: the  parameter list of the experiments.
        :param tunable_params: Dict[tuple1, tuple2] -- a dictionary indicating the position (tuple1) and range (tuple2) of the tunable prop params.
        :param set_up_funcs: set_up function for the beamline.
        :param step_size: the step_size of each iteration.
        """
        super(OptimParticleSwarm, self).__init__(params, tunable_params, set_up_funcs)
        return


    def init_particles(self, num_particles, velocity_range):
        """
        initiate the particles for PSO.
        :param velocity_range: the range of velocity.
        :return:
        """
        self._initiation()
        # num_particles = len(self.tunable_params_positions)
        self.particles = np.asarray([scale[0] for pos, scale in zip(self.tunable_params_positions, self.tunable_params_ranges)]
                                    ).reshape((1, -1)).repeat(num_particles, axis=0)
        #
        self.global_best_quality = self.prv_quality
        self.global_best_complexity = self._get_reward()[:, -1].mean()
        self.global_best_reward = self.global_best_quality / self.global_best_complexity
        self.global_best_particle = copy.deepcopy(self.particles[0, ])
        #
        self.local_best_rewards = np.ones(shape=(num_particles,)) * self.global_best_reward
        self.local_best_particles = copy.deepcopy(self.particles)
        # We initiate the velocities to go forward to the direction of each tunable parameter.
        self.velocities = 0.5 * np.random.uniform(0, velocity_range, size=(num_particles, len(self.tunable_params_ranges)))
        # self.velocities = 0.1 * np.eye(len(self.tunable_params_positions))
        self.single_step(0.2, 0.0, 0.0, 1.0)
        return

    def single_step(self, inertia_coeff, cognitive_coeff, social_coeff, step_size, velocity_range=None):
        """
        :param inertia_coeff:
        :param cognitive_coeff:
        :param social_coeff:
        :param step_size:
        :param velocity_range:
        :return:
        """
        for n, (local_best_particle, velocity, particle) in enumerate(zip(self.local_best_particles, self.velocities, self.particles)):
            # set the current state to be the globally best pariticle
            self._initiation({pos: self.global_best_particle[i] for i, pos in enumerate(self.tunable_params_positions)})
            # update the particles.
            inertia_term = inertia_coeff * velocity
            cognitive_term = cognitive_coeff * (local_best_particle - particle) #* np.random.uniform(size=particle.shape)
            social_term = social_coeff * (self.global_best_particle - particle) #* np.random.uniform(size=particle.shape)
            velocity_new = inertia_term + cognitive_term + social_term
            # if velocity_range is not None:
            #     velocity_new = np.clip(velocity_new, -velocity_range, velocity_range)
            velocity_new /= np.linalg.norm(velocity_new)
            particle_new = particle + step_size * velocity
            for i, scale in enumerate(self.tunable_params_ranges):
                particle_new[i] = np.clip(particle_new[i], max(self.global_best_particle[i], scale[0]), scale[1])
            #
            self.set_prop_params({pos: particle_new[i] for i, pos in enumerate(self.tunable_params_positions)})
            self._operate_experiments(plot=False)
            #
            rewards = self._get_reward()
            particle_quality = self.global_best_quality + rewards[:, 2].mean()
            particle_complexity = rewards[:, 3].mean()
            particle_reward = particle_quality / particle_complexity
            # set-up the global and local particle record.
            self.particles[n] = particle_new
            self.velocities[n] = velocity_new
            #
            if particle_reward > self.local_best_rewards[n]:
                self.local_best_particles[n] = particle_new
            if particle_reward > self.global_best_reward:
                self.global_best_reward = particle_reward
                self.global_best_particle = particle_new
                self.global_best_quality = particle_quality
                self.global_best_complexity = particle_complexity
        return

    def forward(self, num_particles, velocity_range, inertia_coeff, cognitive_coeff, social_coeff, step_size,
                num_steps=100, early_stopping=1, saveto=None):
        self.init_particles(num_particles, velocity_range)
        best_reward,  bad_steps = self.global_best_reward, 0
        for step in range(num_steps):
            self.single_step(inertia_coeff, cognitive_coeff, social_coeff, step_size, velocity_range)
            print(step, '\t', self.global_best_particle.tolist(), '\t', self.global_best_reward)
            if self.global_best_reward > best_reward:
                best_reward = self.global_best_reward
                bad_steps = 0
            else:
                bad_steps += 1
            if bad_steps >= early_stopping:
                break
        # Print the tunning results.
        print("------------> Finish Optimization.")
        tuned_params = {pos: self.global_best_particle[i] for i, pos in
                        enumerate(self.tunable_params_positions)}
        print(tuned_params)
        self.set_prop_params(tuned_params)
        print("\x1b[1;35mThe number of updates: {}; the best reward: {}.\x1b[0m".format(step, best_reward))
        if saveto:
            save_params(self.other_params, self.prop_params, tuned_params, saveto)
        self._operate_experiments(plot=True)
        return

