#ZQ - This file is for testing combining our versions of both downsizing and density filter

import numpy as np
import utils
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree

def Load_PCD_File(Route):
    """
    Load a PCD file and return points as (3, N) numpy array.
    """
    try:
        pcd = o3d.io.read_point_cloud(Route)
        if not pcd.has_points():
            raise ValueError("PCD has no points")
        Data = np.asarray(pcd.points).T  # (3, N)
        print("Successfully loaded!", "The size of data source is:", Data.shape)
        return Data
    except Exception:
        print("Error! Failed to load file from", Route)
        raise

def o3d_PC_Downsize(pcd_np, size):
    """
    Downsample using open3d voxel grid. Input (3, N), output (3, M).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np.T)
    pcd_ds = pcd.voxel_down_sample(voxel_size=size)
    arr = np.asarray(pcd_ds.points).T
    print("Downsized by o3d:", arr.shape)
    return arr

def Point_Cloud_Downsize(pc_data, Grid_Size):
    """
    Downsample to one representative per voxel.
    Input: pc_data shape (3, N). Output shape (3, M).
    """
    Data_Target = np.asarray(pc_data)
    if Data_Target.shape[0] != 3:
        Data_Target = Data_Target.reshape(-1, 3).T  # fallback

    Cell = np.floor(Data_Target / Grid_Size).T  # shape (N, 3)
    Cell_Indi = []
    Cell_Contend = set()
    for idx, row in enumerate(Cell):
        key = tuple(row)
        if key not in Cell_Contend:
            Cell_Contend.add(key)
            Cell_Indi.append(idx)

    Cell_Indi = np.array(Cell_Indi, dtype=int)
    Result = Data_Target[:, Cell_Indi]  # (3, M)
    print("The Corresponding Data got reduce to the size of: ", Result.shape)
    return Result

def Density_Filter(Source, NN, Ratio):
    Source = np.asarray(Source)
    if Source.shape[0] != 3:
        Source = Source.reshape(-1, 3).T
    Source = Source.T  # KDTree expects (N, 3)
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
    indeices = np.array(Sorted_Data)
    print("After Filtering based on Density, Num of remaining points: ", indeices.shape[0])
    return indeices


def main():
    Source_Data_Routine = '/home/rob422student/Desktop/Final_Proj/CSite1_orig-utm.pcd'
    pcd = Load_PCD_File(Source_Data_Routine)
    
    Source_o3d_Downsized_Arr = o3d_PC_Downsize(pcd, 90)
    Source_Downsized_Arr = Point_Cloud_Downsize(pcd.points, Grid_Size= 80)

    Source_Featured_Arr = Density_Filter(Source_Downsized_Arr, 40, 0.2)

    Source_o3d_Downsized_Vis = Source_o3d_Downsized_Arr.reshape(-1,3,1)
    Source_Downsized_Vis = Source_Downsized_Arr.reshape(-1,3,1)
    Source_Featured_Vis = Source_Featured_Arr.reshape(-1,3,1)
    
    utils.view_pc([Source_o3d_Downsized_Vis, Source_Downsized_Vis], None, ['b','r'], ['o','^'])
    utils.view_pc([Source_Downsized_Vis], None, ['r'], ['^'])
    utils.view_pc([Source_Featured_Vis], None, ['b'], ['o'])
    utils.view_pc([Source_Downsized_Vis, Source_Featured_Vis], None, ['r', 'b'], ['^', 'o'])

    plt.show()



if __name__ == '__main__':
    main()
