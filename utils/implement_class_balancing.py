"""计算类别分布并给出交叉熵损失的权重建议。

该脚本遍历准备好的训练集，统计背景（标签 0）与肋骨（标签 >0）点的数量，
并根据出现频率的倒数生成权重，可传入 ``nn.CrossEntropyLoss(weight=...)``
以缓解肋骨与其他结构之间的严重类别不平衡。

用法::
    python implement_class_balancing.py --root ./dataset/seg_input_10w
"""

import argparse
import os
import numpy as np


def compute_class_weights(root: str):
    """返回训练集的样本数量与基于频率倒数的权重。"""
    train_dir = os.path.join(root, 'train')
    counts = np.zeros(2, dtype=np.int64)  # [背景, 肋骨]
    for fn in os.listdir(train_dir):
        arr = np.load(os.path.join(train_dir, fn))['ct']
        labels = (arr[:, 3] != 0).astype(np.int64)
        unique, cnt = np.unique(labels, return_counts=True)
        counts[unique] += cnt
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return counts, weights


def main():
    parser = argparse.ArgumentParser(description='Derive class weights for rib segmentation')
    parser.add_argument('--root', default=r'D:\CXJ_code\RibSeg\V4\dataset\seg_input_10w', help='Dataset root with train/val/test')
    args = parser.parse_args()

    counts, weights = compute_class_weights(args.root)
    print('Class counts  [background, rib]:', counts.tolist())
    print('Suggested weights for CrossEntropyLoss:', weights.tolist())


if __name__ == '__main__':
    main()