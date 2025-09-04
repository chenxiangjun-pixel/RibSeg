"""检查已生成点云的标签分布。

脚本遍历数据集根目录下的 ``train``、``val`` 与 ``test`` 文件夹，统计
背景与肋骨点的数量，以便在训练前评估类别不平衡情况，并将统计结果存入
Excel 以便进一步分析。

用法::
    python inspect_and_adjust_preprocessing.py --root ./dataset/seg_input_10w
"""

import argparse
import os
import numpy as np
import pandas as pd


def summarize_split(split_dir: str):
    counts = np.zeros(2, dtype=np.int64)  # [背景, 肋骨]
    if not os.path.isdir(split_dir):
        return counts
    for fn in os.listdir(split_dir):
        arr = np.load(os.path.join(split_dir, fn))['ct']
        labels = (arr[:, 3] != 0).astype(np.int64)
        unique, c = np.unique(labels, return_counts=True)
        counts[unique] += c
    return counts


def main():
    parser = argparse.ArgumentParser(description='检查预处理数据集的标签分布')
    parser.add_argument('--root', default='./dataset/seg_input_10w', help='数据集根目录')
    args = parser.parse_args()

    records = []
    for split in ['train', 'val', 'test']:
        counts = summarize_split(os.path.join(args.root, split))
        records.append({'split': split, 'background': int(counts[0]), 'rib': int(counts[1])})
    df = pd.DataFrame(records)
    print(df)
    df.to_excel('label_distribution.xlsx', index=False)
    print('Saved summary to label_distribution.xlsx')


if __name__ == '__main__':
    main()
