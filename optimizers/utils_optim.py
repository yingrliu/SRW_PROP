# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to optimize the propagation parameters.
import copy
from optimizers.utils_reward import *
from optimizers.utils_run import *
from optimizers.utils_operation import *

# # coordinate ascent optimization.
# class _optimizor():
#     def __init__(self, names, setting_params, physics_params, prop_params, index_list, set_up_funcs,
#                 img_path, step_size=0.20, min_range=0.5, max_range=5.0, min_resolution=1.0,
#                 max_resolution=10.0, saveto=None):
#         """
#
#         :param names: list of strings that indicates each instruments.
#         :param setting_params:
#         :param physics_params:
#         :param prop_params:
#         :param index_list:
#         :param set_up_funcs:
#         :param img_path:
#         :param step_size:
#         :param min_range:
#         :param max_range:
#         :param min_resolution:
#         :param max_resolution:
#         :param saveto:
#         """
#         return
#
#     def forward(self):
#         return

def coordinate_ascent(names, setting_params, physics_params, prop_params, index_list, set_up_funcs,
                img_path, step_size=0.20, min_range=0.5, max_range=5.0, min_resolution=1.0, 
                max_resolution=10.0, saveto=None):
    # set the parameters to the minimum.
    index_list = [tuple(index) for index in index_list]
    for index in index_list:
        min_value = min_resolution if index[-1] == 6 or index[-1] == 8 else min_range
        updata_param(prop_params, index, min_value)
    #
    run_experiment(names, setting_params, physics_params, prop_params, set_up_funcs)
    # img_list = glob.glob(os.path.join(img_path, "*.dat"))                  # read the saved experiment results.
    # img_list.remove(os.path.join(img_path, "res_int_se.dat"))
    img_list = [os.path.join(img_path, "res_int_pr_se.dat")]
    # initialize the optimization.
    tune_params = {}
    # todo
    prv_prop_params = copy.deepcopy(prop_params)
    dat_records = [read_dat(path) for path in img_list]
    imgs = [record[0] for record in dat_records]
    mesh_olds = [record[1:] for record in dat_records]
    prev_imgs = [np.zeros_like(img) for img in imgs]
    #
    rewards = np.asarray([get_total_rewards(array_new=img, array_old=prev, mesh_new=mesh_old, mesh_old=mesh_old,
                                                    prv_quality=0, params_new=prv_prop_params, params_list=index_list)
                       for mesh_old, prev, img in zip(mesh_olds, prev_imgs, imgs)])
    global_best_reward = rewards[:, 0].mean()
    prev_imgs, prev_qualities = imgs, rewards[:, 1]
    # check whether the index list is valid.
    for index in index_list:
        if index[-1] < 5 or index[-1] > 8:                         # range.
            raise ValueError("Invalid parameter index!")
    # finetune the range parameter until no parameter can be updated.
    print("------------> Tuning parameters.")
    updates = len(index_list)
    while updates:
        new_updates = 0
        for index in index_list:
            # define new values after update.
            prev_value = prop_params[index[0]][2][index[1]]
            new_value = prop_params[index[0]][2][index[1]] + step_size
            max_value = max_resolution if index[-1] == 5 or index[-1] == 7 else max_range
            min_value = min_resolution if index[-1] == 6 or index[-1] == 8 else min_range
            if new_value < min_value or new_value > max_value:
                continue
            updata_param(prop_params, index, new_value)
            run_experiment(names, setting_params, physics_params, prop_params, set_up_funcs)
            plt.close()
            # compute the retio, reward and complexity.
            dat_records = [read_dat(path) for path in img_list]
            imgs = [record[0] for record in dat_records]
            mesh_news = [record[1:] for record in dat_records]
            #
            rewards = np.asarray([get_total_rewards(array_new=img, array_old=prev, mesh_new=mesh_new, mesh_old=mesh_old,
                                                    prv_quality=prev_quality, params_new=prv_prop_params,
                                                    params_list=index_list)
                                  for mesh_old, mesh_new, prev, img, prev_quality in zip(mesh_olds, mesh_news, prev_imgs, imgs, prev_qualities)])
            current_reward = rewards[:, 0].mean()
            print(index, '\t', new_value, '\t', current_reward)
            if current_reward - global_best_reward > 1e-3:
                # update parameters.
                new_updates += 1
                tune_params[index] = new_value
                global_best_reward = current_reward
                prev_imgs, prev_qualities = imgs, rewards[:, 1]
                mesh_olds = mesh_news
            else:
                # reset the parameter.
                updata_param(prop_params, index, prev_value)
        updates = new_updates
    # visualize the best result.
    print(tune_params)
    run_experiment(names, setting_params, physics_params, prop_params, set_up_funcs, plot=True)
    if saveto:
        save_params(physics_params, prop_params, tune_params, saveto)
    return tune_params

# steepest ascent optimization, each time we only update the param that lead to the largest improvement.
def steepest_ascent(names, setting_params, physics_params, prop_params, index_list, set_up_funcs,
                img_path, step_size=0.20, min_range=0.5, max_range=5.0, min_resolution=1.0,
                max_resolution=10.0, saveto=None):
    # set the parameters to the minimum.
    index_list = [tuple(index) for index in index_list]
    for index in index_list:
        min_value = min_resolution if index[-1] == 6 or index[-1] == 8 else min_range
        updata_param(prop_params, index, min_value)
    #
    run_experiment(names, setting_params, physics_params, prop_params, set_up_funcs)
    # img_list = glob.glob(os.path.join(img_path, "*.dat"))                  # read the saved experiment results.
    # img_list.remove(os.path.join(img_path, "res_int_se.dat"))
    img_list = [os.path.join(img_path, "res_int_pr_se.dat")]
    # initialize the optimization.
    tune_params = {}
    # initialize.
    prv_prop_params = copy.deepcopy(prop_params)
    dat_records = [read_dat(path) for path in img_list]
    imgs = [record[0] for record in dat_records]
    mesh_olds = [record[1:] for record in dat_records]
    prev_imgs = [np.zeros_like(img) for img in imgs]
    #
    rewards = np.asarray([get_total_rewards(array_new=img, array_old=prev, mesh_new=mesh_old, mesh_old=mesh_old,
                                                    prv_quality=0, params_new=prv_prop_params, params_list=index_list)
                       for mesh_old, prev, img in zip(mesh_olds, prev_imgs, imgs)])
    global_best_reward = rewards[:, 0].mean()
    prev_imgs, prev_qualities = imgs, rewards[:, 1]
    # check whether the index list is valid.
    for index in index_list:
        if index[-1] < 5 or index[-1] > 8:                         # range.
            raise ValueError("Invalid parameter index!")
    # finetune the range parameter until no parameter can be updated.
    print("------------> Tuning parameters.")
    updates = True
    while updates:
        updates = False
        update_idx, update_val, best_reward = None, None, global_best_reward
        curr_imgs, curr_quality = imgs, rewards[:, 1]
        for index in index_list:
            # define new values after update.
            prev_value = prop_params[index[0]][2][index[1]]
            new_value = prop_params[index[0]][2][index[1]] + step_size
            max_value = max_resolution if index[-1] == 5 or index[-1] == 7 else max_range
            min_value = min_resolution if index[-1] == 6 or index[-1] == 8 else min_range
            if new_value < min_value or new_value > max_value:
                continue
            updata_param(prop_params, index, new_value)
            run_experiment(names, setting_params, physics_params, prop_params, set_up_funcs)
            plt.close()
            # compute the retio, reward and complexity.
            dat_records = [read_dat(path) for path in img_list]
            imgs = [record[0] for record in dat_records]
            mesh_news = [record[1:] for record in dat_records]
            #
            rewards = np.asarray([get_total_rewards(array_new=img, array_old=prev, mesh_new=mesh_new, mesh_old=mesh_old,
                                                    prv_quality=prev_quality, params_new=prv_prop_params,
                                                    params_list=index_list)
                                  for mesh_old, mesh_new, prev, img, prev_quality in zip(mesh_olds, mesh_news, prev_imgs, imgs, prev_qualities)])
            current_reward = rewards[:, 0].mean()
            updata_param(prop_params, index, prev_value)            # reset the value.
            if current_reward - best_reward > 1e-3:
                # update parameters.
                update_idx, update_val = index, new_value
                updates = True
                best_reward = current_reward
                curr_imgs, curr_qualities = imgs, rewards[:, 1]
                curr_mesh = mesh_news
        # update the parameter that leads to the steepest improvement.
        if updates:
            print(update_idx, '\t', update_val, '\t', best_reward)
            updata_param(prop_params, update_idx, update_val)
            global_best_reward = best_reward
            prev_imgs, prev_qualities, mesh_olds = curr_imgs, curr_qualities, curr_mesh
            tune_params[update_idx] = update_val
    # visualize the best result.
    print(tune_params)
    run_experiment(names, setting_params, physics_params, prop_params, set_up_funcs, plot=True)
    if saveto:
        save_params(physics_params, prop_params, tune_params, saveto)
    return tune_params