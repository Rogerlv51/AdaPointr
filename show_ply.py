import open3d as o3d
import numpy as np

def read_pcd(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def self_normalize(pc):
    min_val = np.min(pc, axis=0)
    max_val = np.max(pc, axis=0)
    range_val = max_val - min_val
    point_cloud_normalized = pc - min_val
    point_cloud_normalized /= range_val
    point_cloud_normalized -= 0.5
    return point_cloud_normalized


if __name__ == '__main__':
    path = '2023-07-27_11_39_57-Teeth-11.ply'
    point_cloud = read_pcd(path)
    pc_1 = pc_norm(point_cloud)
    pc_2 = self_normalize(point_cloud)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_2)

    o3d.io.write_point_cloud("point_cloud.ply", pcd)
