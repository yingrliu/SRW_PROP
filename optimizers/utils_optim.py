# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to optimize the propagation parameters.
import copy
from optimizers.utils_reward import *
from optimizers.utils_run import *
from optimizers.utils_operation import *


# coordinate ascent optimization.
class _optimizer():
    def __init__(self, names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs,
                 *args, **kwargs):
        """

        :param names: list of strings that indicates each instruments.
        :param setting_params: the general setting parameters of the experiments.
        :param physics_params: the physical parameters of the experiments.
        :param prop_params: the propagation parameters of the experiments.
        :param tunable_params: Dict[tuple1, tuple2] -- a dictionary indicating the position (tuple1) and range (tuple2) of the tunable prop params.
        :param set_up_funcs:
        :param img_path: path to save the output figures of the experiments.
        :param saveto: path to save the optimal values of the propagation parameters.
        """
        # TODO: add assert.
        self.names, self.setting_params, self.physics_params, self.prop_params = names, setting_params, physics_params, prop_params
        self.tunable_params_positions, self.tunable_params_ranges = [], []
        for position, value in tunable_params.items():
            # check whether the index list is valid.
            if position[-1] < 5 or position[-1] > 8:  # range.
                raise ValueError("Invalid parameter index!")
            self.tunable_params_positions.append(position)
            self.tunable_params_ranges.append(value)
        self.reset_prop_params()
        # save the parameters to run the experiments.
        self.cache_path, self.set_up_func = self.setting_params[1][2], set_up_funcs
        run_experiment(names, setting_params, physics_params, prop_params, set_up_funcs)
        # img_list = glob.glob(os.path.join(img_path, "*.dat"))                  # read the saved experiment results.
        # img_list.remove(os.path.join(img_path, "res_int_se.dat"))
        self.result_files = [os.path.join(self.cache_path, "res_int_pr_se.dat")]
        self.arrays, self.prv_arrays, self.prv_prop_params, self.mesh, self.prv_mesh = None, None, None, None, None
        self.prv_quality = 0.
        return

    def forward(self):
        return

    def reset_prop_params(self, init_values=None):
        """

        :param init_values: Dict[tuple1, float] -- a dictionary indicating the position (tuple1) and value (float) of the tunable prop params.
        :return:
        """
        # TODO: add assertion.
        if init_values is not None:
            for position, value in init_values.items():
                updata_param(self.prop_params, position, value)
        else:
            # initiate the tunable parameters to its smallest value.
            for position, value in zip(self.tunable_params_positions, self.tunable_params_ranges):
                updata_param(self.prop_params, position, value[0])
        return

    def _operate_experiments(self, plot=False):
        run_experiment(self.names, self.setting_params, self.physics_params, self.prop_params, self.set_up_func,
                       plot=plot)
        if not plot:
            plt.close()
        # compute the retio, reward and complexity.
        dat_records = [read_dat(path) for path in self.result_files]
        self.arrays = [record[0] for record in dat_records]
        self.mesh = [record[1:] for record in dat_records]
        return

    def _get_reward(self):
        rewards = np.asarray([get_total_rewards(array_new=current, array_old=prev, mesh_new=mesh, mesh_old=mesh_old,
                                                prv_quality=self.prv_quality, params_new=self.prop_params)
                              for mesh_old, mesh, prev, current in zip(self.prv_mesh, self.mesh, self.prv_arrays, self.arrays)])
        return rewards



class Coordinate_Ascent(_optimizer):
    def __init__(self, names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs, step_size=0.20):
        super(Coordinate_Ascent, self).__init__(names, setting_params, physics_params, prop_params, tunable_params,
                                                set_up_funcs)
        self.step_size = step_size
        return

    def forward(self, saveto=None):
        tuned_params = {}
        # Initiate the optimization process.
        self.reset_prop_params()
        self._operate_experiments(plot=False)
        self.prv_mesh = copy.deepcopy(self.mesh)
        self.prv_arrays = [np.zeros_like(array) for array in self.arrays]
        rewards = self._get_reward()
        global_best_reward = rewards[:, 0].mean()
        self.prv_arrays, self.prv_quality = copy.deepcopy(self.arrays), rewards[:, 1]
        # Optimization process.
        print("------------> Tuning parameters.")
        updates = len(self.tunable_params_positions)
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
                rewards = self._get_reward()
                current_reward = rewards[:, 0].mean()
                print(position, '\t', new_value, '\t', current_reward)
                # check whether there is improvement.
                if current_reward - global_best_reward > 1e-3:
                    # update parameters.
                    new_updates += 1
                    tuned_params[position] = new_value
                    global_best_reward = current_reward
                    self.prv_arrays, self.prv_mesh, self.prv_quality = copy.deepcopy(self.arrays), \
                                                                       copy.deepcopy(self.mesh), rewards[:, 1]
                else:
                    # reset the parameter.
                    updata_param(self.prop_params, position, prev_value)
            updates = new_updates
        # Print the tunning results.
        print("------------> Finish Optimization.")
        print(tuned_params)
        self._operate_experiments(plot=True)
        if saveto:
            save_params(self.physics_params, self.prop_params, tuned_params, saveto)
        return tuned_params



class Reinforce(_optimizer):
    def __init__(self, names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs,
                 step_size=0.20):
        super(Reinforce, self).__init__(names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs)
        #
        self.MultiModelNet = None
        return

    def train(self):
        """
        the process the train the neural network by REINFORCE
        :return:
        """
        return

    def forward(self):
        """
        used the trained neural network to optimize the propagation parameters.
        :return:
        """
        return



class ActorCrtics(_optimizer):
    pass