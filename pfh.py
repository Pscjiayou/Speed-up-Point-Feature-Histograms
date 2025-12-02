import numpy as np


'''
function: point_neighbors(X, ratio)
find the neighbors of each point

input: X: 3xN np.arry/matrix: all points matrix, 
       ratio: float: ratio of points to be considered as neighbors

output: closest_indices: Nxk np.arry/matrix: row i is the indices of all neighbors of point i
'''
def point_neighbors(X, ratio):
    k = int(ratio * X.shape[1])     # number of points considered as neighboors
    if k < 3:
        k = 3

    dists = np.sum(np.square(X[:, :, None] - X[:, None, :]), axis = 0)
    closest_indices = np.argsort(dists, axis=1)[:, 1: k + 1]
    return closest_indices.astype(int)


'''
function: surface_normals(X, ratio)
compue the normal vector of all points according to its neighbors

input: X: 3xN np.arry/matrix: all points matrix, 
       ratio: float: ratio of points to be considered as neighbors

output: normal_matrix: 3xn normal vector np.array/matrix with order same as selected points
'''
def surface_normals(X, ratio):
    closest_indices = point_neighbors(X, ratio)
    closest_points = np.take(X, closest_indices, axis=1)
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
function: compute_all_features(X, closest_indices, normal_matrix)
compute all the features for each point, and reduce the computation time

input: X: 3xN np.array/matrix: all point sets
       closest_indices: Nxk np.arry/matrix: row i is the indices of all neighbors of point i
       normal_matrix: 3xn normal vector np.array/matrix with order same as selected points
       
output: features: nx3 np.array/matrix with columns (alpha. phi, theta)
'''
def compute_all_features(X, closest_indices, normal_matrix):
    N = X.shape[1]
    features_dict = {}

    for i in range(N):
        neighbors_indices = closest_indices[i, :]
        is_smaller = (neighbors_indices < i).any()
        if is_smaller:
            smaller_indices = neighbors_indices[np.where(neighbors_indices < i)] # find the pairs that have already computed
            neighbors_indices = neighbors_indices[np.where(neighbors_indices > i)]

        features = point_features(i, neighbors_indices, X, normal_matrix)

        if is_smaller:
            for j in smaller_indices:
                j = int(j)
                temp_features = features_dict[j][np.where(closest_indices[j, :] == i), :]
                if temp_features.size == 0:
                    temp_features = pair_features(X[:, i], 
                                                  X[:, j], 
                                                  normal_matrix[:, i], 
                                                  normal_matrix[:, j])
                features = np.vstack((temp_features.reshape((1, 3)), features))

        features_dict[i] = features

    return features_dict


'''
histogram seperation
'''


'''
compute the correspondence
'''

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