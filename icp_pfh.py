#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import pfh
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target
    #pc_target = utils.load_pc('cloud_icp_target1.csv')
    #pc_target = utils.load_pc('cloud_icp_target2.csv')
    #pc_target = utils.load_pc('cloud_icp_target3.csv')
    # print(pc_source)
    # utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    # plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    pc_source = utils.convert_pc_to_matrix(pc_source)
    pc_target = utils.convert_pc_to_matrix(pc_target)


    '''
    Parameters for different target
    target 0:
        max_iteration = 15
        error_bound = 1e-2

    target 1:
        max_iteration = 15
        error_bound = 1e-3

    target 2:
        max_iteration = 30
        error_bound = 1e-2

    target 3:
        max_iteration = 20
        error_bound = 1e-2
    '''
    max_iteration = 50
    error_bound = 1e-2

    error_list = []
    runs = 0

    p = pc_source
    p = numpy.asarray(p)
    pc_target = numpy.asarray(pc_target)

    while runs < max_iteration:

        # using Euclidean to find the closest point
        distance_square = numpy.sum(numpy.square(p[:, :, None] - pc_target[:, None, :]), axis = 0)
        closest_indices = numpy.argmin(distance_square, axis=1)
        q = pc_target[:, closest_indices]

        p_bar = numpy.mean(p, axis = 1).reshape((3, 1))
        q_bar = numpy.mean(q, axis = 1).reshape((3, 1))
        x = p - p_bar
        y = q - q_bar
        S = x @ y.T
        U, Sigma, Vt = numpy.linalg.svd(S)

        R = Vt.T @ numpy.diag(numpy.array([1, 1, numpy.linalg.det(Vt.T @ U.T)])) @ U.T
        t = q_bar - R @ p_bar

        p = R @ p + t

        error = numpy.sum(numpy.square(numpy.linalg.norm(p - q, axis = 0)))
        error_list.append(error)
        if error < error_bound:
            break

        runs += 1

    print(error)
    x = numpy.arange(1, len(error_list) + 1, 1)
    plt.plot(x, error_list, label="error", color="blue", linestyle="-", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Error v.s. Iteration")

    plt.legend()
    plt.show()

    input("Press enter for next test:")
    plt.close()

    p = utils.convert_matrix_to_pc(numpy.asmatrix(p))
    pc_target = utils.convert_matrix_to_pc(numpy.asmatrix(pc_target))
    utils.view_pc([p, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
