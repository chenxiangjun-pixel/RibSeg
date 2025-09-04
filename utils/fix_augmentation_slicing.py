"""对所有坐标点进行数据增强。该示例展示了正确的切片方式 ``pc[:, :, :3]``，从而对批次中的每个点都执行抖动与缩放，而不是仅增强前三个点。
用法::python fix_augmentation_slicing.py
"""

import numpy as np
from data_utils import data_aug


def augment_point_cloud(pc: np.ndarray) -> np.ndarray:
    """返回 ``pc`` 的增强副本。

    参数
    ------
    pc : np.ndarray
        形状为 ``(B, N, 3)`` 的批量点云数组。
    """
    pc = pc.copy()
    pc[:, :, :3] = data_aug.jitter_point_cloud(pc[:, :, :3], 0.005, 0.01)
    pc[:, :, :3] = data_aug.random_scale_point_cloud(pc[:, :, :3], 0.9, 1.1)
    return pc


def main():
    dummy = np.zeros((2, 30000, 3), dtype=np.float32)
    augmented = augment_point_cloud(dummy)
    print('Input shape:', dummy.shape)
    print('Augmented shape:', augmented.shape)


if __name__ == '__main__':
    main()