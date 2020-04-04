# Author: Yingru Liu
# Institute: Stony Brook University
# auxiliary functions to optimize the propagation parameters.
import glob, os, copy
import numpy as np
from utils_reward import *
from utils_run import *
from utils_operation import *
# todo:
# def grid_search(names, setting_params, physics_params, prop_params, index_list, 
#                 num_points, img_path, min_range=0.5, max_range=5.0, min_resolution=1.0, 
#                 max_resolution=10.0):
#     # split the range and resolution parameters.
#     index_list = [tuple(index) for index in index_list]
#     tune_params = {}
#     range_index_list, resolution_index_list = [], []
#     img_list, prev_prop_params = None, None
#     for index in index_list:
#         if index[-1] in (5, 7):                         # range.
#             range_index_list.append(index)
#         elif index[-1] in (6, 8):
#             resolution_index_list.append(index)
#         else:
#             raise ValueError("Invalid parameter index!")
#     # finetune the range parameter first.
#     print("------------> Tuning range parameters.")
#     for index in range_index_list:
#         # img_ratios_range = []
#         best_reward, value = 0, 0
#         v = np.linspace(min_range, max_range, num_points)
#         for i in range(num_points):
#             update_range(prop_params, index, v[i])
#             run_experiment(names, setting_params, physics_params, prop_params)
#             if img_list is None:
#                 img_list = glob.glob(os.path.join(img_path, "*.dat"))                  # read the saved experiment results.
#                 img_list.remove(os.path.join(img_path, "res_int_se.dat"))
#             imgs = [read_dat(path)[0] for path in img_list]
#             img_reward = sum([get_image_ratio_reward(img, index) for img in imgs])
#             print(index, '\t', v[i], '\t', img_reward)
#             if img_reward >= best_reward:
#                 best_reward, value = img_reward, v[i]
#         print("---------------> Tuning range parameters at ", index, '\tas\t', value,  '.')
#         update_range(prop_params, index, value)
#         tune_params[index] = value
#     # finetune the range parameter first.
#     print("------------> Tuning resolution parameters.")
#     for index in resolution_index_list:
#         # img_ratios_resolution = []
#         best_reward, value = 0, 1.0
#         prev_imgs, prev_quality = None, 0
#         v = np.linspace(min_resolution, max_resolution, num_points)
#         for i in range(num_points):
#             update_resolution(prop_params, index, v[i])
#             run_experiment(names, setting_params, physics_params, prop_params)
#             imgs = [read_dat(path)[0] for path in img_list]
#             if prev_imgs is None:
#                 prev_imgs = [np.zeros_like(img) for img in imgs]
#             quality = np.mean([get_difference(prev, img) for prev, img in zip(prev_imgs, imgs)]) + prev_quality
#             complexity = get_complexity(prop_params)
#             img_reward = quality / complexity
#             print(index, '\t', v[i], '\t', img_reward)
#             if img_reward > best_reward:
#                 best_reward, value = img_reward, v[i]
#             # save current results.
#             prev_imgs, prev_quality = imgs[:], quality
#         print("---------------> Tuning resolution parameters at ", index, '\tas\t', value,  '.')
#         update_resolution(prop_params, index, value)
#         tune_params[index] = value
#     print(tune_params)
#     return

def grid_search(names, setting_params, physics_params, prop_params, index_list, 
                num_points, img_path, min_range=0.5, max_range=5.0, min_resolution=1.0, 
                max_resolution=10.0):
    # split the range and resolution parameters.
    index_list = [tuple(index) for index in index_list]
    tune_params = {}
    img_list, prev_prop_params = None, None
    prev_imgs, prev_quality = None, 0
    global_best_reward = 0
    # check whether the index list is valid.
    for index in index_list:
        if index[-1] < 5 or index[-1] > 8:                         # range.
            raise ValueError("Invalid parameter index!")
    # finetune the range parameter first.
    print("------------> Tuning parameters.")
    for index in index_list:
        if index[-1] == 5 or index[-1] == 7:    
            v = np.linspace(min_range, max_range, num_points)
        else:
            v = np.linspace(min_resolution, max_resolution, num_points)
        best_reward, value = global_best_reward, 0
        best_quality = 0
        for i in range(num_points):
            updata_param(prop_params, index, v[i])
            run_experiment(names, setting_params, physics_params, prop_params)
            # check the image list after running the experiment.
            if img_list is None:
                img_list = glob.glob(os.path.join(img_path, "*.dat"))                  # read the saved experiment results.
                img_list.remove(os.path.join(img_path, "res_int_se.dat"))
            imgs = [read_dat(path)[0] for path in img_list]
            # compute the retio, reward and complexity.
            if prev_imgs is None:
                prev_imgs = [np.zeros_like(img) for img in imgs]
            img_ratios = np.mean([get_image_ratio_reward(img, index) for img in imgs])
            #
            if prev_prop_params is None:
                prev_prop_params = copy.deepcopy(prop_params)
            quality = np.mean([get_difference(prev, img, prev_prop_params, prop_params) for prev, img in zip(prev_imgs, imgs)]) + prev_quality
            complexity = get_complexity(prop_params)
            img_reward = img_ratios * quality / complexity
            print(index, '\t', v[i], '\t', img_reward)
            if img_reward >= best_reward:
                best_reward, value = img_reward, v[i]
                best_quality = quality
            # save current results.
            prev_imgs, prev_quality, prev_prop_params = imgs[:], quality, copy.deepcopy(prop_params)
        print("---------------> Tuning range parameters at ", index, '\tas\t', value,  '.')
        updata_param(prop_params, index, value)
        tune_params[index] = value
        global_best_reward, prev_quality, prev_prop_params = best_reward, best_quality, copy.deepcopy(prop_params)
    print(tune_params)
    return

# todo: performance is worse than grid search.
def coordinate_ascent(names, setting_params, physics_params, prop_params, index_list, 
                img_path, step_size=0.20, min_range=0.5, max_range=5.0, min_resolution=1.0, 
                max_resolution=10.0):
    # set the parameters to the minimum.
    index_list = [tuple(index) for index in index_list]
    for index in index_list:
        min_value = min_resolution if index[-1] == 6 or index[-1] == 8 else min_range
        updata_param(prop_params, index, min_value)
    run_experiment(names, setting_params, physics_params, prop_params)
    img_list = glob.glob(os.path.join(img_path, "*.dat"))                  # read the saved experiment results.
    img_list.remove(os.path.join(img_path, "res_int_se.dat"))
    #
    tune_params = {}
    prev_prop_params = copy.deepcopy(prop_params)
    prev_imgs, prev_quality = [read_dat(path)[0] for path in img_list], 0
    global_best_reward, global_best_ratio = 0, 0
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
            if index[-1] == 5 or index[-1] == 7:
                new_values = [prop_params[index[0]][2][index[1]] + step_size, 
                              prop_params[index[0]][2][index[1]] - step_size]
            else:
                new_values = [prop_params[index[0]][2][index[1]] + step_size]
            for new_value in new_values:
                max_value = max_resolution if index[-1] == 5 or index[-1] == 7 else max_range
                min_value = min_resolution if index[-1] == 6 or index[-1] == 8 else min_range
                if new_value < min_value or new_value > max_value:
                    continue
                updata_param(prop_params, index, new_value)
                run_experiment(names, setting_params, physics_params, prop_params)
                # check the image list after running the experiment.
                imgs = [read_dat(path)[0] for path in img_list]
                # compute the retio, reward and complexity.
                img_ratios = np.mean([get_image_ratio_reward(img, index) for img in imgs])
                #
                quality = np.mean([get_difference(prev, img, prev_prop_params, prop_params) for prev, img in zip(prev_imgs, imgs)]) + prev_quality
                complexity = get_complexity(prop_params)
                img_reward = quality / complexity
                print(index, '\t', new_value, '\t', img_reward, '\t', img_ratios)
                # todo:
                if index[-1] == 5 or index[-1] == 7:                # range.
                    if img_ratios > global_best_ratio:
                        new_updates += 1
                        tune_params[index] = new_value
                        global_best_reward, global_best_ratio, prev_quality = img_reward, img_ratios, quality
                        prev_imgs = imgs
                        break
                else:
                    if img_reward > global_best_reward:
                        new_updates += 1
                        tune_params[index] = new_value
                        global_best_reward, global_best_ratio, prev_quality = img_reward, img_ratios, quality
                        prev_imgs = imgs
                        break
                # reset the parameter.    
                updata_param(prop_params, index, prev_value)
        updates = new_updates
    print(tune_params)
    return