#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path


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
    pc_source = utils.load_pc((base_dir / cfg["pc_source"]).as_posix())
    pc_target = utils.load_pc((base_dir / cfg["pc_target"]).as_posix())

    pc_source = utils.convert_pc_to_matrix(pc_source)
    pc_target = utils.convert_pc_to_matrix(pc_target)

    max_iteration = cfg["max_iteration"]
    error_bound = cfg["error_bound"]

    error_list = []
    runs = 0

    p = np.asarray(pc_source)
    pc_target = np.asarray(pc_target)

    # timing
    start = time.time()

    while runs < max_iteration:

        # using Euclidean distance square to compute the correspondence
        distance_square = np.sum(np.square(p[:, :, None] - pc_target[:, None, :]), axis = 0)
        closest_indices = np.argmin(distance_square, axis=1)
        q = pc_target[:, closest_indices]

        p_bar = np.mean(p, axis = 1).reshape((3, 1))
        q_bar = np.mean(q, axis = 1).reshape((3, 1))
        x = p - p_bar
        y = q - q_bar
        S = x @ y.T
        U, Sigma, Vt = np.linalg.svd(S)

        R = Vt.T @ np.diag(np.array([1, 1, np.linalg.det(Vt.T @ U.T)])) @ U.T
        t = q_bar - R @ p_bar

        p = R @ p + t

        error = np.sum(np.square(np.linalg.norm(p - q, axis = 0)))
        error_list.append(error)
        if error < error_bound:
            break

        runs += 1

    duration = time.time() - start

    print("Final error:", error)
    print("Total run number:", runs)
    print("Total computation time:", round(duration, 4), "s")

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
    pc_source_pc = utils.convert_matrix_to_pc(np.asmatrix(pc_source))
    utils.view_pc([pc_source_pc, pc_target, p], None,
                  cfg.get("colors", ['#8ecae6', '#8bc34a', '#fcba03']),
                  cfg.get("markers", ['o', '^', 's']))


    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
