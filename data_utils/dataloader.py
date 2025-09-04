# *_*coding:utf-8 *_*
import os
import torch
import numpy as np
from torch.utils.data import Dataset

# 如安装了 scikit-learn，则使用其 KDTree 进行高效的邻域搜索。
try:
    from sklearn.neighbors import KDTree
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - 当缺少 sklearn 时退回此分支
    _HAS_SKLEARN = False


def ball_query(points_query, points_ref, K=16, radius=0.05):
    """为每个查询点在给定半径内寻找最多 K 个邻居。

    参数
    ------
    points_query : (N, 3) 数组
        查询点集合。
    points_ref : (M, 3) 数组
        被搜索的参考点云。
    K : int, 可选
        返回的最大邻居数，默认为 16。
    radius : float, 可选
        归一化坐标下的搜索半径，默认为 0.05。

    返回
    ------
    np.ndarray
        形状为 ``(N, K, 3)`` 的邻居坐标，缺失的邻居以零填充。
    """

    if _HAS_SKLEARN:
        tree = KDTree(points_ref)
        dist, idx = tree.query(points_query, k=K, return_distance=True)
        neighbors = points_ref[idx]
        neighbors[dist > radius] = 0.0
        return neighbors

    # 退回方案：使用暴力搜索（速度慢，但无需额外依赖）
    neighbors = np.zeros((points_query.shape[0], K, 3), dtype=points_ref.dtype)
    for i, p in enumerate(points_query):
        dists = np.linalg.norm(points_ref - p, axis=1)
        nn_idx = np.argsort(dists)[:K]
        mask = dists[nn_idx] <= radius
        neighbors[i, mask] = points_ref[nn_idx][mask]
    return neighbors

def pc_normalize(pc,centroid=None,m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid

    if m is None:        
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))        
    pc = pc / m

    return pc, centroid ,m

def ARPD(pc_full, pc_ds, npoints_abs=1000, npoints_relative=29, radius=0.05):
    """构建绝对与相对位置描述符。

    参数
    ------
    pc_full : (N, 3) 数组
        原始点云。
    pc_ds : (M, 3) 数组
        下采样后的输入点坐标。

    返回
    ------
    np.ndarray
        拼接后的数组，形状为 ``(M, 3 + 3 * npoints_relative)``，前 3 列为
        归一化坐标，其余列编码邻居点的相对偏移。
    """
    if pc_full.shape[0] > 100000:
        choice = np.random.choice(pc_full.shape[0], 100000, replace=False)
        pc_full = pc_full[choice]

    point_bq = ball_query(pc_ds, pc_full, K=npoints_relative, radius=radius)
    idx_zero = np.argwhere((point_bq == 0).all(axis=2))

    point_bq -= pc_ds[:, None, :]
    for dim1, dim2 in idx_zero:
        point_bq[dim1, dim2] = 0

    ct = np.concatenate((pc_ds, point_bq.reshape(pc_ds.shape[0], -1)), axis=1)
    return ct

class RibSegDataset(Dataset):
    def __init__(self,root , npoints=30000, split='train', flag_cl = False, flag_arpe = False):
        self.npoints = npoints
        self.root = root
        self.flag_cl = flag_cl
        self.flag_arpe = flag_arpe

        train_ids = set([x for x in os.listdir(self.root+'/train')])
        val_ids = set([x for x in os.listdir(self.root+'/val')])
        test_ids = set([x for x in os.listdir(self.root+'/test')])

        if split == 'trainval':
            self.datapath = [self.root+'/train/'+fn for fn in train_ids]+[self.root+'/val/'+fn for fn in val_ids]
        elif split == 'train':
            self.datapath = [self.root+'/train/'+fn for fn in train_ids]
        elif split == 'val':
            self.datapath = [self.root+'/val/'+fn for fn in val_ids]
        elif split == 'test':
            self.datapath = [self.root+'/test/'+fn for fn in test_ids]
        else:
            self.datapath=[]
            print('未知的数据集划分: %s. 退出...' % (split))
            exit(-1)
    
    def __getitem__(self, index):
        fn = self.datapath[index]
        data = np.load(fn)
        pc = data['ct'].astype(np.float32)
        pc[:,:3], centroid, m = pc_normalize(pc[:,:3])
        
        ct, label = pc[:,:3], pc[:,3]
        
        choice = np.random.choice(ct.shape[0],self.npoints,replace=False)
        ct,label = ct[choice],label[choice]
        
        if self.flag_arpe:
            ct = ARPD(pc[:,:3],ct)

        if self.flag_cl:
            cl = data['cl'].astype(np.float32)
            cl = cl.reshape(-1, 3)
            idx_list = []
            for i in range(24):
                if cl[i * 500][0] != -1:
                    idx_list.append(i)
            idx = []
            for x in idx_list:
                for a in range(x * 500, x * 500 + 500):
                    idx.append(a)
            if idx:
                cl[idx], _, _ = pc_normalize(cl[idx], centroid, m)
            cl = cl.reshape(-1, 500, 3)
            return ct, label, cl
        return ct, label

    def __len__(self):
        return len(self.datapath)
