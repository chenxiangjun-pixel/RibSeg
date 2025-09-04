#!/usr/bin/env python3
"""将点云数据合并为训练/验证/测试子集。

该脚本读取 ``data_prepare.py`` 生成的 ``data_pn`` 与 ``label_pn`` 目录，
将每个病例的坐标与标签数组拼接为单个 ``.npz`` 文件，保存到
``dataset/seg_input_10w/{train,val,test}`` 下。

样本所属的数据集划分由 ``ribfrac-train-info.csv`` 指定，其中包含
``public_id``（如 ``RibFrac301``）和 ``label_code`` 字段；``label_code``
与子集的映射关系为 ``0->train``、``1->val``、``2->test``。
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def merge_split(csv_path: str, source_dir: str = "./data/pn", out_root: str = "./dataset/seg_input_10w") -> None:
    info = pd.read_csv(csv_path)
    split_map = {0: "train", 1: "val", 2: "test"}

    for split in split_map.values():
        os.makedirs(os.path.join(out_root, split), exist_ok=True)

    for _, row in info.iterrows():
        raw_id = str(row["public_id"])
        case_id = raw_id if raw_id.startswith("RibFrac") else f"RibFrac{int(raw_id):03d}"
        data_path = os.path.join(source_dir, "data_pn", f"{case_id}.npy")
        label_path = os.path.join(source_dir, "label_pn", f"{case_id}.npy")
        if not (os.path.isfile(data_path) and os.path.isfile(label_path)):
            # 若缺少对应的点云或标签文件则跳过
            continue

        coords = np.load(data_path)
        labels = np.load(label_path)
        ct = np.c_[coords, labels]
        split = split_map.get(int(row.get("label_code", 0)), "train")
        out_path = os.path.join(out_root, split, f"{case_id}.npz")
        np.savez_compressed(out_path, ct=ct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge point clouds into dataset splits")
    parser.add_argument("--csv", default="ribfrac-train-info.csv", help="Path to info CSV")
    parser.add_argument("--source", default="./data/pn", help="Directory containing data_pn and label_pn")
    parser.add_argument("--out", default="./dataset/seg_input_10w", help="Output root directory")
    args = parser.parse_args()

    merge_split(args.csv, args.source, args.out)
