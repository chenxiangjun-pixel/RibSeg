#!/usr/bin/env python3
"""Merge point-cloud numpy arrays into train/val/test splits.

This utility reads ``data_pn`` and ``label_pn`` directories produced by
``data_prepare.py``. For each case, it combines the coordinate and label arrays
into a single ``.npz`` file under ``dataset/seg_input_10w/{train,val,test}``.

The mapping of samples to splits is defined by ``ribfrac-train-info.csv`` whose
columns include ``public_id`` (e.g. ``RibFrac301``) and ``label_code``. ``label_code`` values are mapped to
splits as ``0->train``, ``1->val`` and ``2->test``.
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
            # skip cases without prepared arrays
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