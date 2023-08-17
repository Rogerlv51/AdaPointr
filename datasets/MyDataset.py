import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

import open3d as o3d

def mesh_to_points(pc, nums):

    # 加载网格文件
    mesh = o3d.io.read_triangle_mesh(pc)
    
    # 根据网格的表面积进行采样
    pcd = mesh.sample_points_uniformly(number_of_points=nums)

    return np.array(pcd.points).astype(np.float32)

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class Teeth(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset

        # Load the dataset indexing file
        # 读取数据集json文件，定义好的点云类别id，训练和测试的文件夹目录名称
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        
        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):    ## 数据transform处理
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:   ## 测试的时候少了一步RandomMirrorPoints的操作
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]   # 划分训练集测试集

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id': s,
                    'partial_path': self.partial_points_path % (subset, dc['taxonomy_id'], s),
                    'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s)
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='MYDATASET')
        return file_list

    def _normalize(self, pc):
        min_val = np.min(pc, axis=0)
        max_val = np.max(pc, axis=0)
        range_val = max_val - min_val
        point_cloud_normalized = pc - min_val
        point_cloud_normalized /= range_val
        point_cloud_normalized -= 0.5
        return point_cloud_normalized
    
    def pc_norm(self, pc):   # TEST：添加shapenet处理归一化方式
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        ## 个人感觉这里相当于shuffle的操作
        rand_idx = 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            if ri == 'partial':
                data[ri] = IO.get(file_path).astype(np.float32)   # TEST：输入输出分开做处理
            else:
                data[ri] = mesh_to_points(file_path, 4096)

            # 这里自己做归一化处理，为了和pcn数据集对齐 TEST：和shapenet对齐
            data[ri] = self._normalize(data[ri])    # 还是必须做归一化处理不然会出问题，同时验证了partial和completion分开做归一化并不会影响分布，还是在同一坐标系
            # 先采样到统一的点数
            # data[ri] = torch.from_numpy(data[ri])
            # 注意在这里最远点采样，不能直接用pointnet的fps函数，因为支持cuda，而我们这里的数据是cpu的
            # data[ri] = farthest_point_sample(data[ri], 2048)
            # data[ri] = data[ri].numpy().astype(np.float32)

        assert data['gt'].shape[0] == self.npoints   # 这里判断，gt对应的点数必须和config里面规定的一致

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)