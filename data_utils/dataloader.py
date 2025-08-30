import os
import torch
import numpy as np
from torch.utils.data import Dataset


# Optional efficient neighbor search if scikit-learn is available.
try:
    from sklearn.neighbors import KDTree
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - fallback when sklearn missing
    _HAS_SKLEARN = False


def ball_query(points_query, points_ref, K=16, radius=0.05):
    """Find up to K neighbors within ``radius`` for each query point.

    Parameters
    ----------
    points_query : (N, 3) array
        Query points.
    points_ref : (M, 3) array
        Reference point cloud to search.
    K : int, optional
        Maximum number of neighbors to return, by default 16.
    radius : float, optional
        Search radius in normalized units, by default 0.05.

    Returns
    -------
    np.ndarray
        Neighbor points of shape ``(N, K, 3)``. Missing neighbors are
        filled with zeros.
    """

    if _HAS_SKLEARN:
        tree = KDTree(points_ref)
        dist, idx = tree.query(points_query, k=K, return_distance=True)
        neighbors = points_ref[idx]
        neighbors[dist > radius] = 0.0
        return neighbors

    # Fallback: brute-force search (slow but dependency-free)
    neighbors = np.zeros((points_query.shape[0], K, 3), dtype=points_ref.dtype)
    for i, p in enumerate(points_query):
        dists = np.linalg.norm(points_ref - p, axis=1)
        nn_idx = np.argsort(dists)[:K]
        mask = dists[nn_idx] <= radius
        neighbors[i, mask] = points_ref[nn_idx][mask]
    return neighbors


def pc_normalize(pc, centroid=None, m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid

    if m is None:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m

    return pc, centroid, m


def ARPD(pc_full, pc_ds, npoints_abs=1000, npoints_relative=29, radius=0.05):
    """Construct absolute and relative position descriptors.

    Parameters
    ----------
    pc_full : (N, 3) array
        Raw point cloud.
    pc_ds : (M, 3) array
        Sub-sampled input point positions.

    Returns
    -------
    np.ndarray
        Concatenated array of shape ``(M, 3 + 3 * npoints_relative)`` where the
        first three columns are the normalized coordinates and the remaining
        columns encode relative offsets to neighboring points.
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
    def __init__(self, root, npoints=30000, split='train', flag_cl=False, flag_arpe=False):
        self.npoints = npoints
        self.root = root
        self.flag_cl = flag_cl
        self.flag_arpe = flag_arpe

        train_ids = set([x for x in os.listdir(self.root + '/train')])
        val_ids = set([x for x in os.listdir(self.root + '/val')])
        test_ids = set([x for x in os.listdir(self.root + '/test')])

        if split == 'trainval':
            self.datapath = [self.root + '/train/' + fn for fn in train_ids] + [self.root + '/val/' + fn for fn in val_ids]
        elif split == 'train':
            self.datapath = [self.root + '/train/' + fn for fn in train_ids]
        elif split == 'val':
            self.datapath = [self.root + '/val/' + fn for fn in val_ids]
        elif split == 'test':
            self.datapath = [self.root + '/test/' + fn for fn in test_ids]
        else:
            self.datapath = []
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

    def __getitem__(self, index):
        fn = self.datapath[index]
        data = np.load(fn)
        pc = data['ct'].astype(np.float32)
        pc[:, :3], centroid, m = pc_normalize(pc[:, :3])

        ct, label = pc[:, :3], pc[:, 3]

        choice = np.random.choice(ct.shape[0], self.npoints, replace=False)
        ct, label = ct[choice], label[choice]

        if self.flag_arpe:
            ct = ARPD(pc[:, :3], ct)

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
