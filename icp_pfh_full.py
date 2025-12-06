#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
import pfh_utils
import time
import json
from pathlib import Path
import Test_Combine


def load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def main():
    # load config
    cfg_path = Path(__file__).with_name("config_icp_pfh.json")
    cfg = load_config(cfg_path)

    # Import the cloud from config (relative to script dir)
    base_dir = Path(__file__).parent
    loader_mode = cfg.get("pcd_or_csv", "csv").lower()
    if loader_mode == "pcd":
        pc_source_raw = Test_Combine.Load_PCD_File((base_dir / cfg["pc_source"]).as_posix())
        pc_target_raw = Test_Combine.Load_PCD_File((base_dir / cfg["pc_target"]).as_posix())
    else:
        pc_source_raw = utils.load_pc((base_dir / cfg["pc_source"]).as_posix())
        pc_target_raw = utils.load_pc((base_dir / cfg["pc_target"]).as_posix())
        pc_source_raw = np.asarray(utils.convert_pc_to_matrix(pc_source_raw))
        pc_target_raw = np.asarray(utils.convert_pc_to_matrix(pc_target_raw))

    # parameters from config
    ratio_or_k = cfg["ratio_or_k"]

    use_target_indices = cfg["use_target_indices"]

    for_FPFH = cfg["for_FPFH"]
    which_cloud_for_bin = cfg["which_cloud_for_bin"]
    bins_per_feature = cfg["bins_per_feature"]
    bin_seperating_method = cfg["bin_seperating_method"]
    distance_method = cfg["distance_method"]

    # parameters for ICP
    max_iteration = cfg["max_iteration"]
    error_bound = cfg["error_bound"]

    error_list = []
    runs = 0

    # algorithm starts
    start = time.time()

    # downsampling and feature selection (keep shape (3, N))
    pc_source = Test_Combine.Point_Cloud_Downsize(pc_source_raw, 0.01)
    p = np.asarray(pc_source)
    pc_target = Test_Combine.Point_Cloud_Downsize(pc_target_raw, 0.01)

    if not use_target_indices:
        target_indices_1 = None
        target_indices_2 = None
    else:
        target_indices_1 = Test_Combine.Density_Filter(p, 10, 0.1).astype(int)
        target_indices_2 = Test_Combine.Density_Filter(pc_target, 10, 0.1).astype(int)

    time_feature_selected = time.time()

    # pfh starts
    closest_indices_2, dists_2, target_indices_with_neighbor_2, target_indices_array_2, all_features_2, features_bin_boundary, number_of_total_bin, total_points_histogram_2 = pfh_utils.pfh_target_cloud_bin(pc_target, ratio_or_k, target_indices_2, for_FPFH, bins_per_feature, bin_seperating_method)

    while runs < max_iteration:
        print(f'run{runs} starts')

        # using pfh to compute the correspondence
        cross_point_cloud_closest_indices = pfh_utils.pfh_matching(p, ratio_or_k, target_indices_1, for_FPFH, 
                which_cloud_for_bin, features_bin_boundary_Target = features_bin_boundary, number_of_total_bin_Target = number_of_total_bin,
                all_features_2 = all_features_2, total_points_histogram_2 = total_points_histogram_2, closest_indices_2 = closest_indices_2, dists_2 = dists_2, 
                target_indices_array_2 = target_indices_array_2, target_indices_with_neighbor_2 = target_indices_with_neighbor_2,
                bins_per_feature = bins_per_feature, bin_seperating_method = bin_seperating_method, distance_method = distance_method)

        p_sub = p[:, cross_point_cloud_closest_indices[:, 0]]
        q_sub = pc_target[:, cross_point_cloud_closest_indices[:, 1]]

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

        error = np.sum(np.square(np.linalg.norm(p - pc_target, axis = 0)))
        error_list.append(error)
        if error < error_bound:
            break

        runs += 1

    end = time.time()

    
    print('The final error is:', error)
    print('The total run number is:', runs)
    print('Total computation time:', round(end - start, 4), 's')
    print('Downsampling and features selecting time:', round(time_feature_selected - start, 4), 's')
    print('Registration time:', round(end - time_feature_selected, 4), 's')

    x = np.arange(1, len(error_list) + 1, 1)
    plt.plot(x, error_list, label="error", color="blue", linestyle="-", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Error v.s. Iteration")

    plt.legend()
    plt.show()

    input("Press enter for next test:")
    plt.close()

    p = utils.convert_matrix_to_pc(np.asmatrix(p))
    pc_target = utils.convert_matrix_to_pc(np.asmatrix(pc_target))
    pc_source = utils.convert_matrix_to_pc(np.asmatrix(pc_source))
    utils.view_pc([pc_source, pc_target, p], None,
                  cfg.get("colors", ['#8ecae6', '#8bc34a', '#fcba03']),
                  cfg.get("markers", ['o', '^', 's']))


    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    print("start icp_pfh")
    main()
