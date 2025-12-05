from nt import close
import numpy as np


'''
key variables passing by:
target_indices
target_indices_with_neighbor
'''


'''
function: point_neighbors(X, ratio)
find the neighbors of each point

input: X: 3xN np.arry/matrix: all points matrix, 
       ratio_or_k: int or float: 
                                float: ratio of points to be considered as neighbors
                                int: number of points to be considered as neighbors

output: closest_indices: Nxk+1 np.arry/matrix: row i is the indices of all neighbors of point i (including itself)
        dists: Nxk np.arry/matrix: row i is the distances of all neighbors of point i (including itself)
'''
def point_neighbors(X, ratio_or_k):
    # k is the number of points considered as neighboors

    if isinstance(ratio_or_k, int):
        k = ratio_or_k
    elif isinstance(ratio_or_k, float) and ratio_or_k <= 1.0 and ratio_or_k >= 0.0:
        k = int(ratio_or_k * X.shape[1])
    else:
        raise TypeError("Wrong input type of ratio or k")
    

    if k < 3:
        print("Warning: The number of neighbors is too small, using k = 3 automatically.")
        k = 3

    dists = np.linalg.norm(X[:, :, None] - X[:, None, :], axis = 0)
    closest_indices = np.argsort(dists, axis=1)[:, : k + 1]

    return closest_indices.astype(int), dists[closest_indices]


'''
compute the indices of targets and their neighbors and safe computation by making them unique
function: unique_targets_and_neighbors_indices(closest_indices, target_indices = None)

input: closest_indices: Nxk+1 np.arry/matrix: row i is the indices of all neighbors of point i (including itself)
       target_indices: None or list or tuple or nd array: indices of key target points
       for_FPFH: bool: used to specify whether the algorithm will use for_FPFH or not
                       if False, target_indices_with_neighbor is target_indices_array 
                                (do not consider neighbors' histogram features)

output: target_indices_with_neighbor: 1d array
                                      1. if target_indices is None: it will return the unique array of 
                                         points used to compute features (targets and their neighbors)
                                      2. if target_indices is not None: it will return unique array of points 
                                         but with the first severl elements to be the target points indices
        target_indices_array: 1d array or None: array version of the indices of key target points
'''
def unique_targets_and_neighbors_indices(closest_indices, target_indices = None, for_FPFH = False):
    N = closest_indices.shape[0]

    if target_indices is not None:
        if not (isinstance(target_indices, np.ndarray)):
            try:
                target_indices_array = np.array(target_indices)
            except:
                raise TypeError("target_indices must be a ndarray or list or tuple.")

        if not np.issubdtype(target_indices.dtype, np.integer):
            raise TypeError("target_indices must be an int.")

        if np.any(target_indices < 0 or target_indices > N - 1):
            raise ValueError("indices must be in the range of number of points")

        if for_FPFH:
            neighbors = closest_indices[target_indices_array, :].flatten()
            target_indices_with_neighbor = np.unique(neighbors)
        else:
            target_indices_with_neighbor = target_indices_array

    else:
        target_indices_with_neighbor = np.arange(N)

    return target_indices_with_neighbor, target_indices_array


'''
function: surface_normals(X, ratio)
compue the normal vector of all points according to its neighbors

input: X: 3xN np.arry/matrix: all points matrix, 
       closest_indices: Nxk+1 np.arry/matrix: row i is the indices of all neighbors of point i (including itself)

output: normal_matrix: 3xn normal vector np.array/matrix with order same as selected points
'''
def surface_normals(X, closest_indices):
    closest_points = np.take(X, closest_indices[:, 1:], axis=1)
    closest_points = np.concatenate([closest_points, X[:, :, None]], axis=2)

    normal_matrix = np.zeros_like(X)

    for i in range(closest_points.shape[1]):          # loop over n
        mat = np.asarray(closest_points[:, i, :])     # shape (3, k+1)
        row_means = mat.mean(axis=1, keepdims=True)
        mat = mat - row_means
        Y = (mat).T / np.sqrt(mat.shape[1] - 1)
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        normal_matrix[:, i] = Vt[2, :]
        
    return normal_matrix


'''
pairwise features computation
function: pair_features(pt, ps, normal_pt, normal_ps)
compute the features of a pair points

input: pt: (3,) np.arry/matrix, target point
       ps: (3,) np.arry/matrix, neighboor point
       normal_pt: (3,) np.arry/matrix, normal vector of target point
       normal_ps: (3,) np.arry/matrix, normal vector of neighboor point

output: features: 1x3 np.array with elements (alpha. phi, theta)
'''
def pair_features(pt, ps, normal_pt, normal_ps):
    diff = pt - ps
    d = np.linalg.norm(diff)
    u = normal_ps
    v = np.cross(u, diff/d)
    w = np.cross(u, v)

    n = normal_pt
    alpha = v @ n
    phi = u @ diff/d
    theta = np.arctan2(w @ n, u @ n)

    return np.array([alpha, phi, theta]).reshape((1, 3))


'''
vectorized version features computation
function: point_features(pt_index, X, neighbors_indices)
compute the pair-wise features of a point with all neighbors

input: pt_index: int: target point index, 
       neighbors_indices: (K,) np.array: all neighbors indices
       X: 3xN np.array/matrix: all point sets
       normal_matrix: 3xn normal vector np.array/matrix with order same as selected points
       
output: features: nx3 np.array/matrix with columns (alpha. phi, theta)
'''
def point_features(pt_index, neighbors_indices, X, normal_matrix):
    neighbors = X[:, neighbors_indices]
    pt = X[:, pt_index].reshape((3, 1))
    diff =  pt - neighbors
    d = np.linalg.norm(diff, axis = 0)
    u = normal_matrix[:, neighbors_indices]
    diff_d = diff/d

    v = np.cross(u.T, diff_d.T).T
    w = np.cross(u.T, v.T).T

    n = normal_matrix[:, pt_index].reshape((3, 1))
    alpha = (v.T @ n).reshape(-1)
    phi = np.diag(u.T @ diff_d).reshape(-1)
    theta = np.arctan2((w.T @ n).reshape(-1), (u.T @ n).reshape(-1))

    features = np.column_stack([alpha, phi, theta])
    return features


'''
function: compute_all_features(X, closest_indices, normal_matrix, target_indices_with_neighbor = None)
compute all the features for each point, and reduce the computation time

input: X: 3xN np.array/matrix: all point sets
       closest_indices: N x k + 1 np.arry/matrix: row i is the indices of all neighbors of point i (including itself)
       normal_matrix: 3 x k normal vector np.array/matrix with order same as selected points
       target_indices_with_neighbor: array of ints: target point indecies and their neighbors, 
                                    if None, compute all points

output: features_dict: k x 3 np.array/matrix with columns (alpha. phi, theta) for each target point
        all_features: number of target and their neighbor points x n x 3 np.array/tensoor 
        with axis 0: target index; (axis 2: (alpha. phi, theta))
'''
def compute_all_features(X, closest_indices, normal_matrix, target_indices_with_neighbor = None):
    N = X.shape[1]

    closest_indices_exclusive = closest_indices[:, 1:]
    k = closest_indices_exclusive.shape[1]

    if target_indices_with_neighbor is not None:
        indices = target_indices_with_neighbor
    else:
        indices = np.arange(N)

    target_num = indices.shape[0]

    all_features = np.zeros((target_num, k, 3))
    features_dict = {}

    for point_index_in_list in range(target_num):
        i = indices[point_index_in_list]
        neighbors_indices = closest_indices_exclusive[i, :]
        is_smaller = (neighbors_indices < i).any()
        if is_smaller:
            smaller_indices = neighbors_indices[np.where(neighbors_indices < i)] # find the pairs that have already computed
            neighbors_indices = neighbors_indices[np.where(neighbors_indices > i)]

        features = point_features(i, neighbors_indices, X, normal_matrix)

        if is_smaller:
            for j in smaller_indices:
                load_features = False

                j = int(j)
                if j in target_indices_with_neighbor:
                    temp_features = features_dict[j][np.where(closest_indices_exclusive[j, :] == i), :]
                    if temp_features.size != 0:
                        load_features = True
                    

                if j not in target_indices_with_neighbor or not load_features:
                    temp_features = pair_features(X[:, i], 
                                                  X[:, j], 
                                                  normal_matrix[:, i], 
                                                  normal_matrix[:, j])

                features = np.vstack((temp_features.reshape((1, 3)), features))

        features_dict[i] = features
        all_features[point_index_in_list, :, :] = features

    return features_dict, all_features


'''
Compute the features' bins boundaries
function: histogram_separation(all_points_features, bins_per_feature = 4, method = 'equal_width')
pass in points' features and return the bin boundaries for each feature (allow inequal bin numbers)

input: all_points_features: number of target points x n x 3 np.array/matrix: see function 
                            compute_all_features's output all_features
       bins_per_feature: int or int list or int ndarray: the number of bins for each feature:
                         if pass in int, there will be equal number of bins for each feature
                         if pass in list or ndarray, the size should be the same as the number of features
       method: str: option (equal_width, percentile)

output: features_bin_boundary: list of bin boundaries used by each of feature 
                                (number of bins + 1 as input -np.inf and np.inf)
        number_of_total_bin: total number of bins combination
'''
def histogram_separation(total_points_features, bins_per_feature = 4, method = 'equal_width'):
    number_features = total_points_features.shape[2]

    if isinstance(bins_per_feature, int):
        is_equal_bin_number = True
        if bins_per_feature <=0:
            raise ValueError("Number of bins must be non-negative")

        number_of_total_bin = number_features * bins_per_feature

    elif isinstance(bins_per_feature, list):
        if all(isinstance(item, int) for item in bins_per_feature):
            is_equal_bin_number = False
            number_of_total_bin = 0
            for i in bins_per_feature:
                if i <=0 :
                    raise ValueError("Number of bins must be non-negative")

                number_of_total_bin += i

        else:
            raise TypeError("Wrong bins_per_feature type!!! \n Pass in int/int list/int ndarray")

    elif isinstance(bins_per_feature, np.ndarray) and np.issubdtype(bins_per_feature.dtype, np.integer):
        is_equal_bin_number = False
        if np.any(bins_per_feature < 0):
            raise ValueError("Number of bins must be non-negative")

        number_of_total_bin = int(np.prod(bins_per_feature))

    else:
        raise TypeError("Wrong bins_per_feature type!!! \n Pass in int/int list/int ndarray")

    features_bin_boundary = []

    # compute the bin boundary
    if method not in ['equal_width', 'percentile']:
        raise ValueError("Wrong method, you should use 'equal_width' or 'percentile'")

    for i in range(number_features):
        features = total_points_features[:, :, i]

        if is_equal_bin_number:
            bin_number = bins_per_feature
        else:
            bin_number = int(bins_per_feature[i])

        if method == 'equal_width':
            max_feature_value = np.max(features)
            min_feature_value = np.min(features)
            gap = (max_feature_value - min_feature_value) / bin_number

            middle_boundary = np.array([min_feature_value + i * gap for i in range(1, bin_number)])
            

        if method == 'percentile':
            gap = 100 / bin_number
            q = [gap * i for i in range(1, bin_number)]

            middle_boundary = np.percentile(features, q)

        features_bin_boundary.append(np.concatenate(([-np.inf], middle_boundary, [np.inf])))

    return features_bin_boundary, number_of_total_bin


'''
Compute the robust histogram for a point and its neighbors
function: histogram_computation_SPFH(all_points_features, features_bin_boundary, number_of_total_bin)
pass in points' features and features' bins boundaries return the frequency histogram 
    (adding an error term for numerical feasibility)

input: total_points_features: number of target and their neighbor points x n x 3 np.array/matrix: 
                                see function compute_all_features's output all_features
       features_bin_boundary: list of bin boundaries used by each of feature 
                                (number of bins + 1 as input -np.inf and np.inf)
       number_of_total_bin: total number of bins combination

output: total_points_histogram: (number of target and neighbors points, total bins number) np.array/matrix 
                                with values to be the percentage of bin combinations
'''
def histogram_computation_SPFH(total_points_features, features_bin_boundary, number_of_total_bin):
    number_points = total_points_features.shape[0]
    number_neighbors = total_points_features.shape[1]

    total_points_histogram = []


    for point in range(number_points):
        point_features = total_points_features[point]
        hist, edges = np.histogramdd(point_features, bins=features_bin_boundary)
        robust_histogram = (hist.ravel() + 1e-8) / (number_neighbors + number_of_total_bin * 1e-8)
        total_points_histogram.append(robust_histogram)

    return np.array(total_points_histogram)


'''
Compute the mixture of histograms for a point and its neighbors (FPFH)
histogram_computation_FPFH(total_points_features, closest_indices, closest_distance_matrix, 
                            target_indices_array, target_indices_with_neighbor)
computing FPFH based on SPFH (see paper:  Fast Point Feature Histograms (FPFH) for 3D Registration)

input: total_points_features: (number of target points, total bins number) np.array/matrix with values to be 
                                the percentage of bin combinations
       closest_indices: N x k+1 np.arry/matrix: row i is the indices of all neighbors of point i 
                        (including itself) (see point_neighbors)
       closest_distance_matrix: N x k np.arry/matrix: row i is the distances of all neighbors of point i 
                                (including itself) (see point_neighbors)
       target_indices_array: (see the output of function unique_targets_and_neighbors_indices)
       target_indices_with_neighbor: (see the output of function unique_targets_and_neighbors_indices)


output: FPFH: (number of target points, total bins number) np.array/matrix with FPFH values of bin combinations
'''
def histogram_computation_FPFH(total_points_features, closest_indices, closest_distance_matrix, 
                                target_indices_array, target_indices_with_neighbor):

    dists_weights = closest_distance_matrix[target_indices_array, :].copy()
    dists_weights[:, 0] = np.ones(target_indices_array.shape[0])

    dists_weights = 1 / dists_weights

    indices = closest_indices[target_indices_array, :]

    lookup = -np.ones(target_indices_with_neighbor.max() + 1, dtype=int)
    lookup[target_indices_with_neighbor] = np.arange(target_indices_with_neighbor.size)

    mapped_indices = lookup[indices]

    including_neighbor_features  = np.take(total_points_features, mapped_indices, axis=0)

    FPFH = np.sum(dists_weights * including_neighbor_features, axis = 1)

    return FPFH


'''
compute the correspondence
function: coorespondence(all_points_histogram1, all_points_histogram2, method = "l2")

input: all_points_histogram1: number of target points x number of bins: the histogram features of each target points
                              of point cloud 1
       all_points_histogram2: number of target points x number of bins: the histogram features of each target points
                              of point cloud 2
       target_indices_array1: 1d array: the index array of target points of point cloud 1
       target_indices_array2: 1d array: the index array of target points of point cloud 2
       method: str: the measurement of similarity (options: [l1, l2, chi-distance, JSD, cosine])

output: closest_indices: (number of target points in cloud 1, 2) array: each row is 
                        (target point index in cloud 1, the index of the most similar target point in cloud 2)
'''
def coorespondence(all_points_histogram1, all_points_histogram2, 
                    target_indices_array1, target_indices_array2, method = "l2"):

    if method not in ["l1", "l2", "chi-distance", "JSD", "cosine"]:
        raise ValueError("Wrong method. Please use any of [l1, l2, chi-distance, JSD, cosine]")

    # N1 x feature x N2
    vectorize_all_points_histogram1 = all_points_histogram1[:, :, None]
    vectorize_all_points_histogram2 = all_points_histogram2[None, :, :]

    if method == "l1":
        diff_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2
        distance = np.linalg.norm(diff_matrix, ord = 1, axis = 1)
        measure = "min"

    elif method == "l2":
        diff_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2
        distance = np.linalg.norm(diff_matrix, axis = 1)
        measure = "min"

    elif method == "chi-distance":
        eps=1e-8
        diff_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2
        sum_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2 + eps
        distance = 0.5 * np.sum(diff_matrix ** 2 / sum_matrix, axis = 1)
        measure = "min"

    elif method == "JSD":
        log1_2 = np.log(vectorize_all_points_histogram1 / vectorize_all_points_histogram2)
        log2_1 = np.log(vectorize_all_points_histogram2 / vectorize_all_points_histogram1)

        KL_1_2 = np.sum(vectorize_all_points_histogram1 * log1_2, axis = 1)
        KL_2_1 = np.sum(vectorize_all_points_histogram2 * log2_1, axis = 1)

        distance = 0.5 * (KL_1_2 + KL_2_1)
        measure = "min"

    elif method == "cosine":
        distance = all_points_histogram1 @ all_points_histogram2.T
        measure = "max"


    if measure == "max":
        closest_indices = np.argmax(distance, axis=1)
    elif measure == "min":
        closest_indices = np.argmin(distance, axis=1)

        closest_indices = target_indices_array2[closest_indices]
        closest_indices = np.stack((target_indices_array1, closest_indices), axis = 1)

    return closest_indices


def main():

    # --- Step 1: create a synthetic point cloud ---
    N = 20
    X = np.random.rand(3, N)  # 3xN matrix of points in [0,1]^3
    ratio = 0.2               # 20% of points as neighbors

    print("Point cloud shape:", X.shape)

    # --- Step 2: compute neighbors ---
    closest_indices = point_neighbors(X, ratio)
    print("closest_indices shape:", closest_indices.shape)
    print("neighbors of point 0:", closest_indices[0])

    # --- Step 3: compute normals ---
    normal_matrix = surface_normals(X, ratio)
    print("normal_matrix shape:", normal_matrix.shape)
    print("normal of point 0:", normal_matrix[:, 0])

    # --- Step 4: compute features for one point ---
    pt_index = 0
    neighbors_indices = closest_indices[pt_index, :]
    features = point_features(pt_index, neighbors_indices, X, normal_matrix)
    print("features shape (point 0):", features.shape)
    print("first few features for point 0:\n", features[:5])

    # --- Step 5: compute all features ---
    features_dict = compute_all_features(X, closest_indices, normal_matrix)
    print("Number of entries in features_dict:", len(features_dict))
    for key in list(features_dict.keys())[:3]:
        print(f"Point {key} features shape:", features_dict[key].shape)
        print(features_dict[key])

if __name__ == "__main__":
    main()