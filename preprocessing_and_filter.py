#ZQ - This file is for testing using open3d's Outlier Downsizing

import open3d as o3d
import numpy as np
import utils
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

np.random.seed(48105)

# def open3d_Preprocessing(Route, Voxel_Size, NN, Std_Ratio):
#     pcd = o3d.io.read_point_cloud(Route)
#     print("Successfully loaded!")
#     Data = np.asarray(pcd.points)
#     print("original data shape",Data.shape)
#     Data = Data.reshape(-1,3)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(Data)

#     keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
#     print("keypoints shape",len(keypoints.points))

#     voxel_size = Voxel_Size
#     pcd_source_ds = keypoints.voxel_down_sample(voxel_size=voxel_size)
#     cl, ind = pcd_source_ds.remove_statistical_outlier(nb_neighbors=NN,std_ratio=Std_Ratio)
    
#     Source_Feature = np.asarray(cl.points)
#     Source_Feature_p = Source_Feature.reshape(-1,3,1)
#     print(Source_Feature_p.shape)
#     # Result = np.asarray(keypoints.points).reshape(-1,3,1)
#     utils.view_pc([Source_Feature_p], None, ['r'], ['^'])
#     plt.show()
#     Source_Feature = Source_Feature.T
#     return Source_Feature

# This function is developed by us. The main idea is to discretize the entire space
# into small cubics and then combine them into one points. This will reduce the total
# number of point cloud data into smaller size.
# Input: Route - The routine of the source file - pcd file
#        Grid_Size - The size of the grid user would like to descritize
#        pc_source_afterdown - the downsized data clouds by using other method
# Output: Result - Downsized Point Cloud Array (3,N,1)

def Point_Cloud_Grid_Downsize(Route, Grid_Size, pc_source_afterdown):
    pcd_GD = o3d.io.read_point_cloud(Route)
    Original_Data = np.asarray(pcd_GD.points).reshape(-1,3)
    # mean_point = np.mean(Original_Data, axis = 0)
    # Original_Data = Original_Data - mean_point
    Grid_Size = Grid_Size
    Cell = np.floor(Original_Data/Grid_Size)
    Cell_Indi = np.array([])
    Cell_Contend = []
    for i in range(len(Cell)):
        Row = tuple(Cell[i])
        if Row not in Cell_Contend:
            Cell_Indi = np.append(Cell_Indi,i)
            Cell_Contend.append(Row)
    
    Result = np.zeros((len(Cell_Indi),3))
    for i in range(len(Cell_Indi)):
        Result[i] = Original_Data[Cell_Indi[i].astype(int)]

    Result = Result.reshape(-1,3,1)
    # print("The Corresponding Data got reduce by using Grid Size Method: ", Result.shape)

    Other_Result = pc_source_afterdown.copy()
    Other_Result = Other_Result.T
    Other_Result = Other_Result.reshape(-1,3,1)
    print("Showing the difference between Grid Downsizing method vs o3d Voxel Downsizing")
    utils.view_pc([Result, Other_Result], None, ['r', 'b'], ['^', 'o'])
    plt.title("Grid Downsizing vs o3d Voxel Downsizing")
    plt.legend(['Grid Downsizing', 'Voxel Downsizing'])
    plt.show()
    Other_Result = Other_Result.T
    return Result


# This function is developed by us. The main idea is to further filter the downsized 
# point cloud into more "feartured"/"representitive" points. This method first use
# Kd-Tree to calculate N nearest neighbor of each point and find the mean distance 
# among those neighbors to determine its sparseness. Then, calculate the global 
# distribution to find the threshold.
# Threshold = global mean + Sigma * global distribution
# Input: Source - Array of data (3,N)
#        NN - The nearest neightbor use would like to compare for Kd-Tree
#        Ratio - Sigma user would like to have
# Output: Result - Filtered Ferature Point Cloud Array (3,N)

def Density_Filter(Source, NN, Ratio):
    Source = np.asarray(Source).reshape(-1,3)
    Ordered_Tree = KDTree(Source)
    Distance_Arr, Indi = Ordered_Tree.query(Source, NN)
    Neighbor_Distance_Mean_Arr = np.mean(Distance_Arr, axis = 1)
    mu = np.mean(Neighbor_Distance_Mean_Arr)
    sigma = np.std(Neighbor_Distance_Mean_Arr)
    D_Max = mu + Ratio * sigma
    Sorted_Data = []
    for n in range(len(Neighbor_Distance_Mean_Arr)):
        if Neighbor_Distance_Mean_Arr[n] <= D_Max:
            Sorted_Data.append(n)
    Result = Source[Sorted_Data]
    print("After Filtering based on Density, shape of remaining points: ", Result.shape)
    return Result


# This function utilizes o3d voxel downsampling to uniformaly downsize the total number 
# of point cloud data.
# Input: Route - The routine of the source file - pcd file
#        Voxel_Size - The size of the voxel user would like to descritize
# Output: Result - Downsized Point Cloud Array (3,N)

def open3d_Preprocessing(Route, Voxel_Size):
    pcd = o3d.io.read_point_cloud(Route)
    print("Successfully loaded!")
    Data = np.asarray(pcd.points)
    print("original data shape",Data.shape)
    Data = Data.reshape(-1,3)
    if(Data.shape[0] >= 700):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(Data)

        voxel_size = Voxel_Size
        pcd_source_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_source_np = np.asarray(pcd_source_ds.points).T
        print("Point Cloud after voxel downsampling: ",len(pcd_source_ds.points))
    else:
        print("Original Size is small enough! No need for downsizing!")
        pcd_source_np = Data.T
    return pcd_source_np



# This function calls density filter to first filter the non-featured points.
# Then transform the point cloud to indies compared to the original downsized
# point cloud.
# Input: pcd_arr - Array of data (3,N)
#        NN - The nearest neightbor use would like to compare for Kd-Tree
#        Std_Ratio - Sigma user would like to have
# Output: indices - Indices of corresponding filtered Ferature Point Cloud Array (N,1)

def filtered_indices(pcd_arr, NN, Std_Ratio): # pcd_arr is (3, N)
    # ensure shape (N, 3) for KDTree without reordering data
    pcd_source_ds = np.asarray(pcd_arr).T  # (N, 3)
    Source_Feature = Density_Filter(pcd_source_ds, NN, Std_Ratio)  # (K, 3)
    Source_Feature_p = Source_Feature.reshape(-1,3,1)
    # print(Source_Feature_p.shape)
    # Result = np.asarray(keypoints.points).reshape(-1,3,1)

    # utils.view_pc([Source_Feature_p], None, ['r'], ['^'])
    # plt.title("Point Cloud After Feature Filtering")
    # plt.show()

    # transpose to (3, K) for column-wise matching
    Source_Feature = Source_Feature.T

    indices = []
    for i in range(Source_Feature.shape[1]):
        pt = Source_Feature[:, i]
        mask = np.all(pcd_arr == pt[:, None], axis=0) 
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError(f"target point {pt} not found")
        indices.append(idx[0])
    indices = np.array(indices)

    return indices

# This function utilizes o3d ISS to filter out the featured points.
# Input: Source - Array of data (3,N)
# Output: Source_Feature - Feartured Point Cloud Array (3,N)
def o3d_PC_ISS(Source): # input is (N, 3)
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(Source)

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd_s)
    Source_Feature = np.asarray(keypoints.points).T
    print("After using ISS, number of feature points are: ", Source_Feature.shape[1])
    return Source_Feature

# This function calls o3d ISS filter and then transform the data to corresponding indices.
# Input: Source - Array of data (3,N)
# Output: Source_Feature - Indices of corresponding filtered Ferature Point Cloud Array (N,1)

def ISS_and_indices(pcd_arr): # input is (3, N)
    pcd_arr_T = np.asarray(pcd_arr).T
    Result = o3d_PC_ISS(pcd_arr_T)              # (K, 3)
    Result_vis = Result.reshape(-1,3,1)
    # utils.view_pc([Result_vis], None, ['r'], ['^'])
    # plt.title("Point Cloud After o3d ISS")
    # plt.show()

    indices = []
    for i in range(Result.shape[1]):
        pt = Result[:, i]
        mask = np.all(pcd_arr == pt[:, None], axis=0) 
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError(f"target point {pt} not found")
        indices.append(idx[0])
    indices = np.array(indices)

    return indices

# This function is used when the target point cloud is null so that the program has to affine 
# transform the source point cloud to create the target point cloud. This function is capable
# of translation, rotation and add noise to the input point cloud.
# Input: pcd - Array of data (N,3) or (3,N)
#        translation - Whether use wants translation or not
#        noise_std - Adding noise ratio
# Output: pts.T - The transformed data cloud (3,N)
#         T - The transformation matrix
def Trans(pcd, translation = True, noise_std = 0.01):
    # make a copy to avoid in-place modification of input
    # normalize numpy input to (N, 3) float64
    arr = np.asarray(pcd, dtype=float)


    if arr.ndim != 2:
        raise ValueError("pcd numpy input must be 2D")
    if arr.shape[0] == 3:
        pts = arr.T  # (N,3)
    elif arr.shape[1] == 3:
        pts = arr
    else:
        raise ValueError("pcd numpy input must be shape (3, N) or (N, 3)")
    cl = o3d.geometry.PointCloud()
    cl.points = o3d.utility.Vector3dVector(pts)

    # creating rotation matrix
    angles = np.random.uniform(-np.pi, np.pi, size=3)

    theta_X = angles[0]
    c = np.cos(theta_X)
    s = np.sin(theta_X)
    R_x = np.array([[1, 0, 0],
                    [0, c, -s],
                    [0,s,c]
                    ])

    theta_Y = angles[1]
    c = np.cos(theta_Y)
    s = np.sin(theta_Y)
    R_y = np.array([[c, 0, s],
                    [0, 1, 0],
                    [-s,0,c]
                    ])

    theta_Z = angles[2]
    c = np.cos(theta_Z)
    s = np.sin(theta_Z)
    R_z = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0,0,1]
                    ])
        
    R = R_x @ R_y @ R_z


    T = np.eye(4)
    T[:3,:3] = R

    # creating transformation: bound translation by current cloud bbox size
    if translation:

        arr = np.asarray(pcd)  # (3, N)

        mins = arr.min(axis=1)
        maxs = arr.max(axis=1)
        extents = maxs - mins  # per-axis box size
        tx = np.random.uniform(-extents[0], extents[0])
        ty = np.random.uniform(-extents[1], extents[1])
        tz = np.random.uniform(-extents[2], extents[2])
        T[:3, 3] = [tx, ty, tz]

    cl.transform(T)  # in-place

    pts = np.asarray(cl.points)
    pts += np.random.normal(0, noise_std, size=pts.shape)
    return pts.T, T


def main():
    # pc_source = utils.load_pc('cloud_icp_source.csv')

    Route = '/home/rob422student/Desktop/Final_Proj/CSite1_orig-utm.pcd'
    open3d_Preprocessing(Route, 80, 40, 0.2)
    

if __name__ == "__main__":
    main()

