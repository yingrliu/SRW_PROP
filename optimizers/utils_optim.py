# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to optimize the propagation parameters.
import copy
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from optimizers.utils_reward import *
from optimizers.utils_run import *
from optimizers.utils_operation import *
from optimizers.NN_package.Modules import OpticNet, OpticNet_DDPG, GaussianNLL


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
        if self.cache_path and not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
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

    def _initiation(self, init_values=None):
        # Initiate the optimization process.
        self.reset_prop_params(init_values)
        self._operate_experiments(plot=False)
        self.prv_mesh = copy.deepcopy(self.mesh)
        self.prv_arrays, self.prv_quality = [np.zeros_like(array) for array in self.arrays], 0.
        self.prv_arrays, self.prv_quality = copy.deepcopy(self.arrays), self._get_reward()[:, 1].mean()
        return



class OptimCoordinateAscent(_optimizer):
    def __init__(self, names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs, step_size=0.20):
        super(OptimCoordinateAscent, self).__init__(names, setting_params, physics_params, prop_params, tunable_params,
                                                set_up_funcs)
        self.step_size = step_size
        return

    def forward(self, saveto=None):
        tuned_params = {}
        # Initiate the optimization process.
        self._initiation()
        global_best_reward = self._get_reward()[:, 0].mean()
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
                                                                       copy.deepcopy(self.mesh), rewards[:, 1].mean()
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



class OptimReinforce(_optimizer):
    def __init__(self, names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs,
                 OpticNetConfig, device='cpu', checkpoint=None):
        """

        :param names:
        :param setting_params:
        :param physics_params:
        :param prop_params:
        :param tunable_params:
        :param set_up_funcs:
        :param OpticNetConfig: dictionary that contains all the parameters required for the OpticNet
        :param device:
        """
        super(OptimReinforce, self).__init__(names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs)
        self.NeuralNet = OpticNet(**OpticNetConfig).to(device=torch.device(device))
        self.OpticNetConfig = OpticNetConfig
        # load the trained model.
        if checkpoint:
            print("\x1b[1;35mLoad trained model.\x1b[0m")
            self.NeuralNet.load_state_dict(torch.load(checkpoint))
        return

    def train(self, numSamples, BeamLine_List, numEpisode=100, std=1e-1, lr=1e-2, saveto=None):
        """
        the process the train the neural network by REINFORCE
        :param numSamples: number of propagation parameter samples used for training.
        :param BeamLine_List [List(str)]: the list of beamlines used for training (each string represents a import statement to import configuration).
        :return:
        """
        # Access all the training beamlines as _optimizers.
        def get_beamline(beam):
            exec("from {} import *".format(beam), globals())
            # get tunable_params
            tunable_params = {}
            for item in index_list:
                tunable_params[item] = [0.75, 5.0] if item[-1] in [5, 7] else [1., 5.]
            return _optimizer(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs)
        BeamLine_Optimizers = [get_beamline(beam) for beam in BeamLine_List]
        # Run the trials.
        optimizer = optim.Adam(self.NeuralNet.parameters(), lr=lr)
        best_losses = float('inf')
        for episode in range(numEpisode):
            random.shuffle(BeamLine_Optimizers)
            avg_loss = 0.
            for beamline in BeamLine_Optimizers:
                reward_per_sample, NLL_per_sample, Prop_param_per_sample = [], [], []
                # run the simulation and collect records.
                for step in range(numSamples):
                    beamline._initiation()
                    types, arrays, props, masks = self._get_tensor_batch(beamline=beamline)
                    Props_new, delta_Prop = self.NeuralNet.forward(types, props, arrays)
                    noise = std * torch.randn_like(delta_Prop).to(next(self.NeuralNet.parameters()).device)   # add noise the the paths.
                    noisy_Props_new = (Props_new + noise).detach()
                    # update params and get new experiment results.
                    clamp_Props_new = self._update_prop_from_tensor(noisy_Props_new, beamline)
                    beamline._operate_experiments(plot=False)
                    reward = beamline._get_reward()
                    reward_per_sample.append(reward[:, 0].mean())
                    nll = (GaussianNLL((clamp_Props_new - props).detach(), delta_Prop.squeeze(), std).squeeze() * masks).sum() / masks.sum()
                    # save the records.
                    NLL_per_sample.append(nll)
                    Prop_param_per_sample.append(clamp_Props_new)
                # compute and optimize the neural network.
                V_values = np.asarray(reward_per_sample)
                V_values = (V_values - V_values.mean()) / (V_values.std() + 1e-4)
                optimizer.zero_grad()
                losses = torch.cat([(nll * v).unsqueeze(0) for nll, v in zip(NLL_per_sample, V_values)]).mean()
                losses.backward()
                optimizer.step()
                avg_loss += losses.detach().cpu().numpy()
                print('\x1b[1;35mLoss: {}\x1b[0m'.format(float(losses.detach().cpu().numpy())))
            avg_loss /= len(BeamLine_Optimizers)
            if avg_loss < best_losses:
                best_losses = avg_loss
                if saveto:
                    if not os.path.exists(saveto):
                        os.makedirs(saveto)
                    print('\x1b[1;35mSaved Model.\x1b[0m')
                    torch.save(self.NeuralNet.state_dict(), os.path.join(saveto, 'model_params.pth'))
        return

    def forward(self, saveto=None):
        """
        used the trained neural network to optimize the propagation parameters.
        :return:
        """
        self._initiation()
        types, arrays, props, masks = self._get_tensor_batch()
        Props_new = self.NeuralNet.forward(types, props, arrays)[0]
        # update params and get new experiment results.
        self._update_prop_from_tensor(Props_new)
        self._operate_experiments(plot=True)
        tuned_params = {}
        for position in self.tunable_params_positions:
            tuned_params[position] = self.prop_params[position[0]][2][position[1]]
        if saveto:
            save_params(self.physics_params, self.prop_params, tuned_params, saveto)
        return

    def _get_tensor_batch(self, beamline=None):
        """
        transform the information we can get from SRW into tensor format (which will be used as the input of the neural network)
        :param beamline: [_optimizer] a optimizer for a specific beamline. If is None, use self's own members.
        :return:
        """
        # Process intensity distribution.
        def process_intensities(array):
            height, weight = array.shape
            torch_array = torch.Tensor(array).to(next(self.NeuralNet.parameters()).device).view(1, 1, height, weight)
            torch_array = F.interpolate(torch_array, (self.OpticNetConfig["dimArray"], self.OpticNetConfig["dimArray"]))
            return torch_array
        # Decide which beamline to use.
        arrays = self.arrays if beamline is None else beamline.arrays
        prop_params = self.prop_params if beamline is None else beamline.prop_params
        physics_params = self.physics_params if beamline is None else beamline.physics_params
        # Transform
        arrays = torch.cat([process_intensities(array) for array in arrays], dim=0)
        props = torch.Tensor([item[2][5:9] for item in prop_params]).to(next(self.NeuralNet.parameters()).device)
        props = props.unsqueeze(1)
        # Mask.
        masks = torch.zeros(size=(len(prop_params), 4)).to(next(self.NeuralNet.parameters()).device)
        for pos in self.tunable_params_positions:
            masks[pos[0], pos[1] - 5] = 1.0
        # # Extract physics parameters for each item and decide its type.
        # extract_physics_params_type(physics_params, [item[0] for item in prop_params])
        types = ['Aperture'] * len(prop_params)
        return types, arrays, props, masks

    def _update_prop_from_tensor(self, prop_tensor, beamline=None):
        """
        use the output of neural network to update the parameters in the optimizer (simulator). This function also
        return a new prop_tensor whose values are clamped by the provided ranges.
        :param prop_tensor:
        :param beamline:
        :return:
        """
        props_new = prop_tensor.squeeze().detach().cpu().numpy()
        prop_params = self.prop_params if beamline is None else beamline.prop_params
        tunable_params_positions = self.tunable_params_positions if beamline is None else beamline.tunable_params_positions
        tunable_params_ranges = self.tunable_params_ranges if beamline is None else beamline.tunable_params_ranges
        for pos, interval in zip(tunable_params_positions, tunable_params_ranges):
            # update the mesh, mesh_old, array, array_old and get rewards.
            new_value = np.clip(props_new[pos[0], pos[1] - 5], interval[0], interval[1])
            updata_param(prop_params, pos, float(new_value))
            props_new[pos[0], pos[1] - 5] = new_value
        return torch.Tensor(props_new).to(next(self.NeuralNet.parameters()).device)


# Deep Deterministic Policy Gradient.
class OptimDDPG(OptimReinforce):
    def __init__(self, names, setting_params, physics_params, prop_params, tunable_params, set_up_funcs,
                 BufferConfig, OpticNetConfig, device='cpu', checkpoint=None):
        """

        :param names:
        :param setting_params:
        :param physics_params:
        :param prop_params:
        :param tunable_params:
        :param set_up_funcs:
        :param OpticNetConfig: dictionary that contains all the parameters required for the OpticNet
        :param device:
        """
        super(OptimDDPG, self).__init__(names, setting_params, physics_params, prop_params, tunable_params,
                                        set_up_funcs, OpticNetConfig, device)
        self.NeuralNet = OpticNet_DDPG(**OpticNetConfig).to(device=torch.device(device))
        # load the trained model.
        if checkpoint:
            print("\x1b[1;35mLoad trained model.\x1b[0m")
            self.NeuralNet.load_state_dict(torch.load(checkpoint))
        return

    def train(self, numSamples, BeamLine_List, numEpisode=100, std=1.0, lr=1e-2, alpha=0.8, saveto=None):
        """
        the process the train the neural network by REINFORCE
        :param numSamples: number of propagation parameter samples used for training.
        :param BeamLine_List [List(str)]: the list of beamlines used for training (each string represents a import statement to import configuration).
        :return:
        """
        """
        the process the train the neural network by REINFORCE
        :param numTrajs: number of propagation parameter trajectories used for training.
        :param numSteps: number of steps of each episode (the number of point of the trajectory)
        :param BeamLine_List [List(str)]: the list of beamlines used for training (each string represents a import statement to import configuration).
        :return:
        """
        # Access all the training beamlines as _optimizers.
        def get_beamline(beam):
            exec("from {} import *".format(beam), globals())
            # get tunable_params
            tunable_params = {}
            for item in index_list:
                tunable_params[item] = [0.75, 5.] if item[-1] in [5, 7] else [1., 5.]
            return _optimizer(names, setting_params, physics_params, propagation_params, tunable_params, set_up_funcs)
        BeamLine_Optimizers = [get_beamline(beam) for beam in BeamLine_List]
        # Run the trials.
        actor_optimizer = optim.Adam(self.NeuralNet.actor_params, lr=lr)
        critic_optimizer = optim.Adam(self.NeuralNet.critic_params, lr=lr)
        # Start training.
        best_losses = float('inf')
        for episode in range(numEpisode):
            random.shuffle(BeamLine_Optimizers)
            avg_loss = 0.
            for beamline in BeamLine_Optimizers:
                beamline._initiation()
                # Collect records to train the critic network.
                rewards, deltas_props, critics = [], [], []
                for step in range(numSamples):
                    # todo:
                    beamline._initiation()
                    types, arrays, props, masks = self._get_tensor_batch(beamline=beamline)
                    Props_new, delta_Prop = self.NeuralNet.forward(types, props, arrays)
                    noise = std * torch.randn_like(delta_Prop).to(
                        next(self.NeuralNet.parameters()).device)  # add noise the the paths.
                    noisy_Props_new = (Props_new + noise).detach()
                    # update params and get new experiment results.
                    clamp_Props_new = self._update_prop_from_tensor(noisy_Props_new, beamline)
                    beamline._operate_experiments(plot=False)
                    reward = beamline._get_reward()
                    # collect training batch for critic network.
                    rewards.append(reward[:, 0].mean())
                    clamp_delta_prop = (clamp_Props_new.unsqueeze(1) - props).detach()
                    deltas_props.append(clamp_delta_prop)
                    _, _, critic = self.NeuralNet.forward(types, props, arrays, clamp_delta_prop)
                    critics.append(critic)
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                rewards = np.asarray(rewards)
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
                losses = sum([(reward - critic) ** 2 for reward, critic in zip(rewards, critics)]) / numSamples
                losses.backward()
                critic_optimizer.step()
                print('\x1b[1;35mEpisode {} - Critic Loss: {}\x1b[0m'.format(episode, float(losses.detach().cpu().numpy())))
                # Training the actor network with the critic network.
                beamline._initiation()
                types, arrays, props, masks = self._get_tensor_batch(beamline=beamline)
                Props_new, delta_Prop = self.NeuralNet.forward(types, props, arrays)
                _, _, critic = self.NeuralNet.forward(types, props, arrays, delta_Prop)
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                (-critic).backward()
                actor_optimizer.step()
                print('\x1b[1;35mEpisode {} - Actor Loss: {}\x1b[0m'.format(episode, float(-critic.detach().cpu().numpy())))
                avg_loss += float(-critic.detach().cpu().numpy()) + float(losses.detach().cpu().numpy())
            avg_loss /= len(BeamLine_Optimizers)
            if avg_loss < best_losses:
                best_losses = avg_loss
                if saveto:
                    if not os.path.exists(saveto):
                        os.makedirs(saveto)
                    print('\x1b[1;35mSaved Model.\x1b[0m')
                    torch.save(self.NeuralNet.state_dict(), os.path.join(saveto, 'model_params.pth'))
        return