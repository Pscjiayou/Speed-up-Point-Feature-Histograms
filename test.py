import open3d as o3d
import numpy as np

import pandas as pd


def load_pcd(filename):
    """
    Loads point cloud data from a CSV file (assuming no header and X, Y, Z columns)
    and converts it into an Open3D PointCloud object.<br/>    """
    print(f"Loading data from {filename}...")
    
    # Load the CSV. We use header=None because your original files had no explicit header.
    # We assign placeholder names 'x', 'y', 'z' for clarity.
    try:
        df = pd.read_csv(filename, header=None, names=['x', 'y', 'z'])
    except Exception as e:
        print(f"Error loading CSV {filename}: {e}")
        return o3d.geometry.PointCloud() # Return an empty cloud on error
    
    # Convert the DataFrame columns (X, Y, Z) into a NumPy array
    points = df[['x', 'y', 'z']].values
    
        # Create the Open3D PointCloud object   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"Loaded {len(pcd.points)} points.")
    return pcd
def main():
# --- 1. PROCESS SOURCE CLOUD ---<br/>    
    print("\n--- Processing Source Cloud (cloud_icp_source.csv) ---")
    pcd_source = load_pcd("cloud_icp_source.csv")
    # Downsampling<br/>    
    voxel_size = 0.01
    pcd_source_ds = pcd_source.voxel_down_sample(voxel_size=voxel_size)
    print(f"Source Cloud Downsampled. Points: {len(pcd_source_ds.points)}")
    # --- 3. VISUALIZATION (Added for confirmation) ---<br/>    
    print("\n--- Starting Visualization ---")
    # Color the clouds for visual distinction
    pcd_source_ds.paint_uniform_color([1, 0, 0]) # Red
    
    # Display both downsampled clouds to check their initial relative position<br/>    
    o3d.visualization.draw_geometries([pcd_source_ds],
                                      window_name="Downsampled Source (Red) and Target (Blue)",
                                      zoom=0.8)
    print("Visualization closed.")
    
if __name__ == "__main__":
    main()