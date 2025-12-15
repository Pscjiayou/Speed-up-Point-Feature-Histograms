#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
import pfh_utils
import time
import json
from pathlib import Path
import preprocessing_and_filter

np.random.seed(48105)

def load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def main():
    # load config
    cfg_path = Path(__file__).with_name("config_test.json")
    cfg = load_config(cfg_path)

    # Import the cloud from config (relative to script dir)
    base_dir = Path(__file__).parent

    # grouped configs
    data_cfg = cfg["data_loading"]
    prep_cfg = cfg["preprocessing"]
    sel_cfg = cfg["target_point_selection"]
    pfh_cfg = cfg["pfh_parameters"]
    icp_cfg = cfg["icp_parameters"]
    vis_cfg = cfg["visualization"]

    # parameters for data preprocessing:
    Voxel_Size = prep_cfg["voxel_size"]

    # parameters for target cloud generation:
    translation = prep_cfg["need_translation"]
    noise_level = prep_cfg.get("noise_level", 0.01)

    # parameters for point selection
    NN = sel_cfg["density_number_of_neighbors"]
    Std_Ratio = sel_cfg["density_selection_ratio"]

    # parameters for FPH
    ratio_or_k = pfh_cfg["ratio_or_k"]

    use_target_indices = sel_cfg["use_filter_or_not"]

    for_FPFH = pfh_cfg["for_FPFH"]
    which_cloud_for_bin = pfh_cfg["which_cloud_for_bin"]
    bins_per_feature = pfh_cfg["bins_per_feature"]
    bin_seperating_method = pfh_cfg["bin_seperating_method"]
    distance_method = pfh_cfg["distance_method"]

    # parameters for ICP
    max_iteration = icp_cfg["max_iteration"]
    error_bound = icp_cfg["error_bound"]

    error_list = []
    runs = 0
    T_cumulated = np.eye(4)


    # loading in data and preprocessing
    loader_mode = data_cfg.get("pcd_or_csv", "pcd").lower()
    if loader_mode == "pcd":
        print("------------Start Downsampling!------------------")
        pc_source = preprocessing_and_filter.open3d_Preprocessing((base_dir / data_cfg["pc_source"]).as_posix(), Voxel_Size)
        preprocessing_and_filter.Point_Cloud_Grid_Downsize((base_dir / data_cfg["pc_source"]).as_posix(), Voxel_Size, pc_source)
        if data_cfg["pc_target"] is None:
            pc_target, True_T = preprocessing_and_filter.Trans(pc_source, translation, noise_level)
        else:
            pc_target = preprocessing_and_filter.open3d_Preprocessing((base_dir / data_cfg["pc_target"]).as_posix(), Voxel_Size)
            preprocessing_and_filter.Point_Cloud_Grid_Downsize((base_dir / data_cfg["pc_source"]).as_posix(), Voxel_Size, pc_target)
    else:
        pc_source_raw = utils.load_pc((base_dir / data_cfg["pc_source"]).as_posix())
        pc_source = np.asarray(utils.convert_pc_to_matrix(pc_source_raw))
        if data_cfg["pc_target"] is None:
            pc_target, True_T = preprocessing_and_filter.Trans(pc_source, translation, noise_level)
        else:
            pc_target_raw = utils.load_pc((base_dir / data_cfg["pc_target"]).as_posix())
            pc_target = np.asarray(utils.convert_pc_to_matrix(pc_target_raw))


    # checking whether the number of source and target data point are the same
    if pc_source.shape[1] < pc_target.shape[1]:
        N_ori = pc_target.shape[1]
        Target = pc_source.shape[1]
        column_i = np.random.choice(N_ori, Target, replace=False)
        pc_target = pc_target[:,column_i]
        pc_target = pc_target
    elif pc_source.shape[1] > pc_target.shape[1]:
        N_ori = pc_source.shape[1]
        Target = pc_target.shape[1]
        column_i = np.random.choice(N_ori, Target, replace=False)
        pc_source = pc_source[:,column_i]

    p = np.asarray(pc_source)

    print('Source points number after downsizing:', pc_source.shape)
    print('Target points number after downsizing:', pc_target.shape)

    # algorithm starts
    start = time.time()
    
    # selecting target points for matching
    if not use_target_indices:
        target_indices_1 = None
        target_indices_2 = None
    else:
        if sel_cfg.get("filter", "density_filter") == "density_filter":
            print("------------Start Feature Filtering with Density Filter!------------------")
            # print("Start Showing Feature Filtered for Source:")
            target_indices_1 = preprocessing_and_filter.filtered_indices(pc_source, NN, Std_Ratio).astype(int)
            # print("Start Showing Feature Filtered for Target:")
            target_indices_2 = preprocessing_and_filter.filtered_indices(pc_target, NN, Std_Ratio).astype(int)
        elif sel_cfg.get("filter") == "iss":
            print("------------Start Feature Filtering with Open3d ISS!------------------")
            # print("Start Showing Feature Filtered for Source:")
            target_indices_1 = preprocessing_and_filter.ISS_and_indices(pc_source).astype(int)
            # print("Start Showing Feature Filtered for Target:")
            target_indices_2 = preprocessing_and_filter.ISS_and_indices(pc_target).astype(int)
        else:
            raise ValueError("If you want to use specific target points, please input a filter type.")

        if target_indices_1.shape[0] < target_indices_2.shape[0]:
            N_ori = target_indices_2.shape[0]
            Target = target_indices_1.shape[0]
            column_i = np.random.choice(N_ori, Target, replace=False)
            target_indices_2 = target_indices_2[column_i]
        elif target_indices_1.shape[0] > target_indices_2.shape[0]:
            N_ori = target_indices_1.shape[0]
            Target = target_indices_2.shape[0]
            column_i = np.random.choice(N_ori, Target, replace=False)
            target_indices_1 = target_indices_1[column_i]
        print("\nTotal feature points number for Source: ", target_indices_1.shape)
        print("Total feature points number for Target: ", target_indices_2.shape)
    
   
        Vis_Source = pc_source.T.copy()
        Vis_Source = Vis_Source.reshape(-1,3,1)
        Vis_Source_Feature = pc_source.T[target_indices_1].copy()
        Vis_Source_Feature = Vis_Source_Feature.reshape(-1,3,1)

        Vis_Target = pc_target.T.copy()
        Vis_Target = Vis_Target.reshape(-1,3,1)
        Vis_Target_Feature = pc_target.T[target_indices_2].copy()
        Vis_Target_Feature = Vis_Target_Feature.reshape(-1,3,1)

        




    plt.show()
    time_feature_selected = time.time()

    print("------------Start Match Testing!------------------")
    print(f"It will run {max_iteration} times!")
    # pfh initialization alignment
    if pfh_cfg["use_pfh"]:
        initialization_runs = pfh_cfg["initialization_runs"]
        ratio_init_sampling = pfh_cfg["ratio_init_sampling"]
        
        closest_indices_2, dists_2, target_indices_with_neighbor_2, target_indices_array_2, all_features_2, features_bin_boundary, number_of_total_bin, total_points_histogram_2 = pfh_utils.pfh_target_cloud_bin(pc_target, ratio_or_k, target_indices_2, for_FPFH, bins_per_feature, bin_seperating_method)

        p, T_cumulated, error_0 = pfh_utils.pfh_matching(p, pc_target, ratio_or_k, target_indices_1, for_FPFH, 
                    which_cloud_for_bin, features_bin_boundary_Target = features_bin_boundary, number_of_total_bin_Target = number_of_total_bin,
                    all_features_2 = all_features_2, total_points_histogram_2 = total_points_histogram_2, closest_indices_2 = closest_indices_2, dists_2 = dists_2, 
                    target_indices_array_2 = target_indices_array_2, target_indices_with_neighbor_2 = target_indices_with_neighbor_2,
                    bins_per_feature = bins_per_feature, bin_seperating_method = bin_seperating_method, distance_method = distance_method, initialization_runs = initialization_runs, ratio_init_sampling = ratio_init_sampling)

        error_list.append(error_0)

        initial_alignment_time = time.time()

    while runs < max_iteration:
        print(f'run{runs} starts')

        if use_target_indices:
            p_sub = p[:, target_indices_1]
            q_sub = pc_target[:, target_indices_2]
            distance_square = np.sum(np.square(p_sub[:, :, None] - q_sub[:, None, :]), axis = 0)
            closest_indices = np.argmin(distance_square, axis=1)
            q_sub = q_sub[:, closest_indices]
        else:
            distance_square = np.sum(np.square(p[:, :, None] - pc_target[:, None, :]), axis = 0)
            closest_indices = np.argmin(distance_square, axis=1)
            p_sub = p
            q_sub = pc_target[:, closest_indices]

        # usual ICP part
        p_bar = np.mean(p_sub, axis = 1).reshape((3, 1))
        q_bar = np.mean(q_sub, axis = 1).reshape((3, 1))
        x = p_sub - p_bar
        y = q_sub - q_bar
        S = x @ y.T
        U, Sigma, Vt = np.linalg.svd(S)

        R = Vt.T @ np.diag(np.array([1, 1, np.linalg.det(Vt.T @ U.T)])) @ U.T
        t = q_bar - R @ p_bar

        p = R @ p + t

        error = np.mean(np.linalg.norm(p - pc_target, axis = 0))
        error_list.append(error)

        T_temp = np.eye(4)
        T_temp[:3,:3] = R
        T_temp[:3, 3] = t.ravel()

        T_cumulated = T_temp @ T_cumulated

        if error < error_bound:
            break

        runs += 1

    end = time.time()

    print("\n--------------------------Results!---------------------------------")
    print('The final error is:', error)
    print('The total run number is:', runs)
    print('Total computation time:', round(end - start, 4), 's')
    print('Downsampling and features selecting time:', round(time_feature_selected - start, 4), 's')
    print('Computing matching time:', round(end - time_feature_selected, 4), 's')
    if pfh_cfg["use_pfh"]:
        print('PFH features matching time:', round(initial_alignment_time - time_feature_selected, 4), 's')
        print('ICP matching time', round(end - initial_alignment_time, 4), 's')

    if data_cfg["pc_target"] is None:
        print('Ground True transformation:\n', True_T)
        print('Approximated transformation:\n', T_cumulated)

    x = np.arange(len(error_list))
    plt.plot(x, error_list, label="error", color="blue", linestyle="-", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Error v.s. Iteration")

    plt.legend()
    plt.show()

    input("------------Press enter to get into the visualization part!------------")
    plt.close()

    input("Press enter for visualizing the source:")

    p_pc = utils.convert_matrix_to_pc(np.asmatrix(p))
    pc_target_pc = utils.convert_matrix_to_pc(np.asmatrix(pc_target))
    pc_source_pc = utils.convert_matrix_to_pc(np.asmatrix(pc_source))

    # figure 1: source only
    utils.view_pc([pc_source_pc], None,
                  [vis_cfg.get("colors", ['#8ecae6', '#8bc34a', '#fcba03'])[0]],
                  [vis_cfg.get("markers", ['o', '^', 's'])[0]])
    plt.title("Point Cloud Source Visualization")
    plt.legend(['source'])

    # figure: Showing Feature Points compared to Source and Target
    if use_target_indices:
        input("Press enter for visualizing the featured points selected in the source:")
        plt.close()

        utils.view_pc([Vis_Source, Vis_Source_Feature], None, 
                    ['b','r'], 
                    ['o','^'])
        plt.title("Source Point Cloud vs Feature After Filtering")
        plt.legend(['Source', 'Source Feature'])

        input("Press enter for visualizing the featured points selected in the target:")
        plt.close()

        utils.view_pc([Vis_Target, Vis_Target_Feature], None, 
                    ['g','r'], 
                    ['o','^'])
        plt.title("Source Point Cloud vs Feature After Filtering")
        plt.legend(['Target', 'Target Feature'])

    # figure 2: aligned (p) vs target
    input("Press enter for visualizing the matching:")
    plt.close()

    utils.view_pc([pc_target_pc, p_pc], None,
                  vis_cfg.get("colors", ['#8ecae6', '#8bc34a', '#fcba03'])[1:3],
                  vis_cfg.get("markers", ['o', '^', 's'])[1:3])
    plt.title("Point Cloud Matching Result Visualization")
    plt.legend(['Target', 'Transformed Data'])

    # figure 3: all in one
    input("Press enter for visualizing the all point clouds in one plot:")
    plt.close()

    utils.view_pc([pc_source_pc, pc_target_pc, p_pc], None,
                  vis_cfg.get("colors", ['#8ecae6', '#8bc34a', '#fcba03']),
                  vis_cfg.get("markers", ['o', '^', 's']))
    plt.title("Entire Point Cloud Visualization")
    plt.legend(['Source', 'Target', 'Transformed Data'])

    plt.show()

    print("Thanks for using!\n")



if __name__ == '__main__':
    print("start icp_pfh")
    main()
