import numpy as np


'''
The key parameters:
    ratio_or_k = float (in the range 0 ~ 1) / int larger than 3
    target_indices_1 = None / 1-d array of int / list of int
    target_indices_2 = None / 1-d array of int / list of int
    for_FPFH = True / False
    which_cloud_for_bin = "target" / "source" / "both"
    bins_per_feature = positive int / 1-d array of positive int / list of positive int
    bin_seperating_method = 'equal_width' / 'percentile'
    distance_method = "l1" / "l2" / "chi-distance" / "JSD" / "cosine"
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

    N = X.shape[1]
    dists = np.linalg.norm(X[:, :, None] - X[:, None, :], axis = 0)
    closest_indices = np.argsort(dists, axis=1)[:, : k + 1]

    # distances aligned with closest_indices; shape (N, k+1)
    closest_dists = dists[np.arange(N)[:, None], closest_indices]

    return closest_indices.astype(int), closest_dists


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
    target_indices_array = None

    if target_indices is not None:
        try:
            target_indices_array = np.asarray(target_indices)
        except:
            raise TypeError("target_indices must be a ndarray or list or tuple.")

        if not np.issubdtype(target_indices_array.dtype, np.integer):
            raise TypeError("target_indices must be an int.")

        if np.any((target_indices_array < 0) | (target_indices_array > N - 1)):
            raise ValueError("indices must be in the range of number of points")

        if for_FPFH:
            neighbors = closest_indices[target_indices_array, :].flatten()
            target_indices_with_neighbor = np.unique(neighbors)
        else:
            target_indices_with_neighbor = target_indices_array

    else:
        target_indices_array = np.arange(N)
        target_indices_with_neighbor = target_indices_array

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
       target_indices_with_neighbor: array of ints: target point indecies and their neighbors

output: features_dict: k x 3 np.array/matrix with columns (alpha. phi, theta) for each target point
        all_features: number of target and their neighbor points x n x 3 np.array/tensoor 
        with axis 0: target index; (axis 2: (alpha. phi, theta))
'''
def compute_all_features(X, closest_indices, normal_matrix, target_indices_with_neighbor):
    N = X.shape[1]

    closest_indices_exclusive = closest_indices[:, 1:]
    k = closest_indices_exclusive.shape[1]

    indices = np.asarray(target_indices_with_neighbor, dtype=int)

    target_num = indices.shape[0]

    all_features = np.zeros((target_num, k, 3))
    features_dict = {}
    target_lookup = None if target_indices_with_neighbor is None else set(indices.tolist())

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
                if (target_lookup is None) or (j in target_lookup):
                    previous_features = features_dict.get(j, None)
                    if previous_features is not None:
                        temp_features = previous_features[np.where(closest_indices_exclusive[j, :] == i), :]
                        if temp_features.size != 0:
                            load_features = True
                    

                if (target_lookup is not None and j not in target_lookup) or not load_features:
                    temp_features = pair_features(X[:, i], 
                                                  X[:, j], 
                                                  normal_matrix[:, i], 
                                                  normal_matrix[:, j])

                features = np.vstack((temp_features.reshape((1, 3)), features))

        features_dict[i] = features
        all_features[point_index_in_list, :, :] = features

    # return features_dict, all_features
    return all_features


'''
Compute the features' bins boundaries
function: histogram_separation(all_points_features, bins_per_feature = 4, method = 'equal_width')
pass in points' features and return the bin boundaries for each feature (allow inequal bin numbers)

input: all_points_features: number of target points x n x 3 np.array/matrix: see function 
                            compute_all_features's output all_features
       bins_per_feature: int or int list or int ndarray: the number of bins for each feature:
                         if pass in int, there will be equal number of bins for each feature
                         if pass in list or ndarray, the size should be the same as the number of features
       bin_seperating_method: str: option (equal_width, percentile)

output: features_bin_boundary: list of bin boundaries used by each of feature 
                                (number of bins + 1 as input -np.inf and np.inf)
        number_of_total_bin: total number of bins combination
'''
def histogram_separation(total_points_features, bins_per_feature = 4, bin_seperating_method = 'equal_width'):
    number_features = total_points_features.shape[2]

    if isinstance(bins_per_feature, int):
        is_equal_bin_number = True
        if bins_per_feature <=0:
            raise ValueError("Number of bins must be non-negative")

        number_of_total_bin = bins_per_feature ** number_features

    elif isinstance(bins_per_feature, list):
        if all(isinstance(item, int) for item in bins_per_feature):
            is_equal_bin_number = False
            if len(bins_per_feature) != number_features:
                raise ValueError("bins_per_feature length must match number of features")
            for i in bins_per_feature:
                if i <=0 :
                    raise ValueError("Number of bins must be non-negative")

            number_of_total_bin = int(np.prod(bins_per_feature))

        else:
            raise TypeError("Wrong bins_per_feature type!!! \n Pass in int/int list/int ndarray")

    elif isinstance(bins_per_feature, np.ndarray) and np.issubdtype(bins_per_feature.dtype, np.integer):
        is_equal_bin_number = False
        if bins_per_feature.shape[0] != number_features:
            raise ValueError("bins_per_feature length must match number of features")
        if np.any(bins_per_feature <= 0):
            raise ValueError("Number of bins must be non-negative")

        number_of_total_bin = int(np.prod(bins_per_feature))

    else:
        raise TypeError("Wrong bins_per_feature type!!! \n Pass in int/int list/int ndarray")

    features_bin_boundary = []

    # normalize method string to accept both " " and ' ' and common unicode quotes
    if isinstance(bin_seperating_method, str):
        bin_seperating_method = bin_seperating_method.strip().strip('"').strip("'").strip("“”‘’").lower()
    if bin_seperating_method not in ['equal_width', 'percentile']:
        raise ValueError("Wrong method, you should use 'equal_width' or 'percentile'")

    for i in range(number_features):
        features = total_points_features[:, :, i]

        if is_equal_bin_number:
            bin_number = bins_per_feature
        else:
            bin_number = int(bins_per_feature[i])

        if bin_seperating_method == 'equal_width':
            max_feature_value = np.max(features)
            min_feature_value = np.min(features)
            gap = (max_feature_value - min_feature_value) / bin_number

            middle_boundary = np.array([min_feature_value + i * gap for i in range(1, bin_number)])
            

        if bin_seperating_method == 'percentile':
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

    if np.any(mapped_indices < 0):
        raise ValueError("Found neighbors whose SPFH was not precomputed.")

    including_neighbor_features  = np.take(total_points_features, mapped_indices, axis=0)

    # add trailing axis so weights broadcast to (points, k+1, bins)
    FPFH = np.sum(dists_weights[..., None] * including_neighbor_features, axis = 1)

    return FPFH


'''
compute the correspondence
function: correspondence(all_points_histogram1, all_points_histogram2, method = "l2")

input: all_points_histogram1: number of target points x number of bins: the histogram features of each target points
                              of point cloud 1
       all_points_histogram2: number of target points x number of bins: the histogram features of each target points
                              of point cloud 2
       target_indices_array1: 1d array: the index array of target points of point cloud 1
       target_indices_array2: 1d array: the index array of target points of point cloud 2
       distance_method: str: the measurement of similarity (options: [l1, l2, chi-distance, JSD, cosine])

output: closest_indices: (number of target points in cloud 1, 2) array: each row is 
                        (target point index in cloud 1, the index of the most similar target point in cloud 2)
'''
def correspondence(all_points_histogram1, all_points_histogram2, 
                    target_indices_array1, target_indices_array2, distance_method = "l2"):

    if distance_method not in ["l1", "l2", "chi-distance", "JSD", "cosine"]:
        raise ValueError("Wrong distance_method. Please use any of [l1, l2, chi-distance, JSD, cosine]")

    # N1 x feature x N2  (feature on axis=1)
    vectorize_all_points_histogram1 = all_points_histogram1[:, :, None]           # (N1, F, 1)
    vectorize_all_points_histogram2 = all_points_histogram2.T[None, :, :]        # (1, F, N2)

    if distance_method == "l1":
        diff_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2
        distance = np.linalg.norm(diff_matrix, ord = 1, axis = 1)
        measure = "min"

    elif distance_method == "l2":
        diff_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2
        distance = np.linalg.norm(diff_matrix, axis = 1)
        measure = "min"

    elif distance_method == "chi-distance":
        eps=1e-8
        diff_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2
        sum_matrix = vectorize_all_points_histogram1 - vectorize_all_points_histogram2 + eps
        distance = 0.5 * np.sum(diff_matrix ** 2 / sum_matrix, axis = 1)
        measure = "min"

    elif distance_method == "JSD":
        log1_2 = np.log(vectorize_all_points_histogram1 / vectorize_all_points_histogram2)
        log2_1 = np.log(vectorize_all_points_histogram2 / vectorize_all_points_histogram1)

        KL_1_2 = np.sum(vectorize_all_points_histogram1 * log1_2, axis = 1)
        KL_2_1 = np.sum(vectorize_all_points_histogram2 * log2_1, axis = 1)

        distance = 0.5 * (KL_1_2 + KL_2_1)
        measure = "min"

    elif distance_method == "cosine":
        distance = all_points_histogram1 @ all_points_histogram2.T
        measure = "max"


    if measure == "max":
        closest_indices = np.argmax(distance, axis=1)
    elif measure == "min":
        closest_indices = np.argmin(distance, axis=1)

    closest_indices = target_indices_array2[closest_indices]
    closest_indices = np.stack((target_indices_array1, closest_indices), axis = 1)

    return closest_indices


'''
function: pfh_raw_features_preparation(point_cloud, ratio_or_k, target_indices=None, for_FPFH=False)
prepare PFH-related primitives for a point cloud (neighbors, normals, pairwise features).

input:
    point_cloud: 3xN np.array/matrix: the point cloud
    ratio_or_k: float or int: neighbor ratio or fixed neighbor count
    target_indices: None/list/ndarray: target point indices; if None, use all points
    for_FPFH: bool: if True, include neighbors of targets in the returned index set

output:
    closest_indices: Nx(k+1) int array: neighbor indices (including self)
    dists: Nx(k+1) float array: distances aligned with closest_indices
    target_indices_with_neighbor: 1d int array: unique indices of targets and their neighbors
    target_indices_array: 1d int array: target indices (or all points if target_indices is None)
    all_features: len(target_indices_with_neighbor) x k x 3 array: pairwise features
'''
def pfh_raw_features_preparation(point_cloud, ratio_or_k = 10, target_indices = None, for_FPFH = False):
    closest_indices, dists = point_neighbors(point_cloud, ratio_or_k)
    target_indices_with_neighbor, target_indices_array = unique_targets_and_neighbors_indices(closest_indices, target_indices, for_FPFH)
    normal_matrix = surface_normals(point_cloud, closest_indices)
    all_features = compute_all_features(point_cloud, closest_indices, normal_matrix, target_indices_with_neighbor)
    return closest_indices, dists, target_indices_with_neighbor, target_indices_array, all_features

'''
function: pfh_target_cloud_bin(point_cloud, ratio_or_k, target_indices=None, for_FPFH=False, bins_per_feature=4, bin_seperating_method="equal_width")
compute bin boundaries and SPFH/FPFH histograms for a specific point cloud.

input:
    point_cloud: 3xN np.array/matrix
    ratio_or_k: float/int: neighbor ratio or neighbor count
    target_indices: None or 1d array: target point indices
    for_FPFH: bool: whether to compute FPFH (otherwise only SPFH)
    bins_per_feature: int or int list/ndarray: bins per feature
    bin_seperating_method: 'equal_width' or 'percentile'

output:
    closest_indices_2, dists_2: neighbor indices/distances
    target_indices_with_neighbor_2, target_indices_array_2: target and neighbor indices
    all_features_2: pairwise features tensor
    features_bin_boundary: bin boundaries per feature
    number_of_total_bin: total bin combination count
    total_points_histogram_2: SPFH/FPFH histograms
'''
def pfh_target_cloud_bin(point_cloud, ratio_or_k = 10, target_indices = None, for_FPFH = False, bins_per_feature = 4, bin_seperating_method = "equal_width"):
    closest_indices_2, dists_2, target_indices_with_neighbor_2, target_indices_array_2, all_features_2 = pfh_raw_features_preparation(point_cloud, 
                                                                                                                                        ratio_or_k, target_indices, for_FPFH)
    features_bin_boundary, number_of_total_bin = histogram_separation(all_features_2, bins_per_feature, bin_seperating_method)
    total_points_histogram_2 = histogram_computation_SPFH(all_features_2, features_bin_boundary, number_of_total_bin)

    if for_FPFH:
        total_points_histogram_2 = histogram_computation_FPFH(total_points_histogram_2, closest_indices_2, dists_2, 
                                    target_indices_array_2, target_indices_with_neighbor_2)

    return closest_indices_2, dists_2, target_indices_with_neighbor_2, target_indices_array_2, all_features_2, features_bin_boundary, number_of_total_bin, total_points_histogram_2

'''
function: pfh_matching(point_cloud_1, ratio_or_k, target_indices=None, for_FPFH=False, which_cloud_for_bin="Target", ...)
compute cross-cloud correspondence using PFH/SPFH/FPFH with configurable bin source.

input:
    point_cloud_1: 3xN np.array/matrix (source cloud for correspondence)
    ratio_or_k: neighbor ratio or count
    target_indices: None/1d array: target indices in cloud 1
    for_FPFH: bool: compute FPFH if True; else SPFH
    which_cloud_for_bin: 'target'/'source'/'both': which cloud determines bin boundaries
    features_bin_boundary_Target, number_of_total_bin_Target: precomputed bins (required if which_cloud_for_bin='target')
    all_features_2, total_points_histogram_2, closest_indices_2, dists_2, target_indices_array_2, target_indices_with_neighbor_2:
        required when which_cloud_for_bin is 'source' or 'both'
    bins_per_feature, bin_seperating_method: bin configuration
    distance_method: 'l1'/'l2'/'chi-distance'/'JSD'/'cosine'

output:
    cross_point_cloud_closest_indices: (N1, 2) array, each row is (idx in cloud1, matched idx in cloud2)
'''
def pfh_matching(point_cloud_1, ratio_or_k = 10, target_indices = None, for_FPFH = False, 
                which_cloud_for_bin = "Target", features_bin_boundary_Target = None, number_of_total_bin_Target = None, 
                all_features_2 = None, total_points_histogram_2 = None, closest_indices_2 = None, dists_2 = None, 
                target_indices_array_2 = None, target_indices_with_neighbor_2 = None,
                bins_per_feature = 4, bin_seperating_method = "equal_width", distance_method = "l2"):
    closest_indices_1, dists_1, target_indices_with_neighbor_1, target_indices_array_1, all_features_1 = pfh_raw_features_preparation(point_cloud_1, 
                                                                                                                                        ratio_or_k, target_indices, for_FPFH)
                                                    

    if isinstance(which_cloud_for_bin, str):
        which_cloud_for_bin = which_cloud_for_bin.strip().strip('"').strip("'").strip("“”‘’").lower()
    if which_cloud_for_bin not in ["target", "source", "both"]:
        raise ValueError("which_cloud_for_bin must be one of [Target, Source, Both]")

    if which_cloud_for_bin == "target":
        if features_bin_boundary_Target is None or number_of_total_bin_Target is None:
            raise ValueError("Must pass in features_bin_boundary_Target and number_of_total_bin for using current bin seperation")

        features_bin_boundary = features_bin_boundary_Target
        number_of_total_bin = number_of_total_bin_Target

    elif which_cloud_for_bin == "source":
        if all_features_2 is None:
            raise ValueError("Must pass in all_features_2")
        features_bin_boundary, number_of_total_bin = histogram_separation(all_features_1, bins_per_feature, bin_seperating_method)

        total_points_histogram_2 = histogram_computation_SPFH(all_features_2, features_bin_boundary, number_of_total_bin)

    elif which_cloud_for_bin == "both":
        if all_features_2 is None:
            raise ValueError("Must pass in all_features_2")

        all_features = np.vstack((all_features_2, all_features_1))

        features_bin_boundary, number_of_total_bin = histogram_separation(all_features, bins_per_feature, bin_seperating_method)

        total_points_histogram_2 = histogram_computation_SPFH(all_features_2, features_bin_boundary, number_of_total_bin)


    total_points_histogram_1 = histogram_computation_SPFH(all_features_1, features_bin_boundary, number_of_total_bin)
    

    if for_FPFH:
        total_points_histogram_1 = histogram_computation_FPFH(total_points_histogram_1, closest_indices_1, dists_1, 
                                target_indices_array_1, target_indices_with_neighbor_1)
        
        if which_cloud_for_bin == "source" or which_cloud_for_bin == "both":
            if any(val is None for val in [closest_indices_2, dists_2, target_indices_array_2, target_indices_with_neighbor_2]):
                raise ValueError("Must input all of [closest_indices_2, dists_2, target_indices_array_2, target_indices_with_neighbor_2]")

            total_points_histogram_2 = histogram_computation_FPFH(total_points_histogram_2, closest_indices_2, dists_2, 
                                    target_indices_array_2, target_indices_with_neighbor_2)

    cross_point_cloud_closest_indices = correspondence(total_points_histogram_1, total_points_histogram_2, 
    target_indices_array_1, target_indices_array_2, distance_method)

    return cross_point_cloud_closest_indices

'''
For testing whether the file is damaged or not
'''
def main():

    np.random.seed(0)

    # --- Step 1: create a synthetic point cloud ---
    N = 30
    X = np.random.rand(3, N)  # 3xN matrix of points in [0,1]^3
    ratio_or_k = 0.2          # 20% of points as neighbors

    print("Point cloud shape:", X.shape)

    # --- Step 2: compute neighbors and distances ---
    closest_indices, closest_dists = point_neighbors(X, ratio_or_k)
    print("closest_indices shape:", closest_indices.shape)
    print("neighbors of point 0 (including itself):", closest_indices[0])

    # --- Step 3: prepare target sets (use all points, include neighbors for FPFH) ---
    target_indices_with_neighbor, target_indices_array = unique_targets_and_neighbors_indices(
        closest_indices, target_indices=None, for_FPFH=True
    )

    # --- Step 4: compute normals ---
    normal_matrix = surface_normals(X, closest_indices)
    print("normal_matrix shape:", normal_matrix.shape)
    print("normal of point 0:", normal_matrix[:, 0])

    # --- Step 5: compute pairwise features for all required points ---
    all_features = compute_all_features(
        X, closest_indices, normal_matrix, target_indices_with_neighbor
    )
    print("all_features shape (points, neighbors, features):", all_features.shape)

    # --- Step 6: compute histogram bin boundaries ---
    bins_per_feature = 5
    features_bin_boundary, number_of_total_bin = histogram_separation(
        all_features, bins_per_feature=bins_per_feature, method='percentile'
    )
    print("bins per feature:", bins_per_feature, "total bin combinations:", number_of_total_bin)

    # --- Step 7: compute SPFH for all selected points ---
    spfh = histogram_computation_SPFH(all_features, features_bin_boundary, number_of_total_bin)
    print("SPFH shape:", spfh.shape)

    # --- Step 8: compute FPFH for target points ---
    fpfh = histogram_computation_FPFH(
        spfh, closest_indices, closest_dists, target_indices_array, target_indices_with_neighbor
    )
    print("FPFH shape:", fpfh.shape)
    print("FPFH sample (first target point, first 5 bins):", fpfh[0, :5])

    # --- Basic sanity checks ---
    assert all_features.shape[0] == target_indices_with_neighbor.shape[0]
    assert spfh.shape[0] == target_indices_with_neighbor.shape[0]
    assert fpfh.shape[0] == target_indices_array.shape[0]
    print("All sanity checks passed.")

if __name__ == "__main__":
    main()