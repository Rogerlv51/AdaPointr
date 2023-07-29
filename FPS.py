import numpy as np
# from plyfile import PlyData

import torch


def farthest_point_sample(points, num_samples):
    """
    使用farthest point sampling从点云中采样指定数量的点。
    Args:
        points: (N, 3) tensor，输入点云的坐标。
        num_samples: 采样点的数量。
    Returns:
        new_points: (num_samples, 3) tensor，采样得到的点云坐标。
    """
    N = points.shape[0]
    new_points_idx = torch.zeros(num_samples, dtype=torch.long)
    distances = torch.ones(N, dtype=torch.float32) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).item()

    for i in range(num_samples):
        new_points_idx[i] = farthest
        new_point = points[farthest].view(1, -1)
        dists = torch.sum((points - new_point) ** 2, dim=1)
        mask = dists < distances
        distances[mask] = dists[mask]
        farthest = torch.max(distances, dim=0)[1].item()

    new_points = points[new_points_idx, :]

    return new_points

# def read_data(path):
#     plydata = PlyData.read(path)
#     vertex = np.array([list(x)[:3] for x in plydata['vertex'].data])
#     v_torch = torch.from_numpy(vertex)
#     return v_torch
    
# path = 'dataset/test/ss.ply'
# data = read_data(path=path)

# fps_data_torch = farthest_point_sample(data, 512)

# fps_data_np = fps_data_torch.numpy()

# # 保存numpy数组为.pts文件
# np.savetxt("tttttteestpoints.pts", fps_data_np, delimiter=" ")


