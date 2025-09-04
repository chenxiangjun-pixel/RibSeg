import argparse
import os
import numpy as np
import pandas as pd


def check_npz_files(split_dir: str):
    """检查指定目录下的所有.npz文件，验证标签列中是否存在非零值"""
    if not os.path.isdir(split_dir):
        print(f"目录不存在: {split_dir}")
        return

    file_count = 0
    files_with_nonzero_labels = 0

    for fn in os.listdir(split_dir):
        if not fn.endswith('.npz'):
            continue

        file_path = os.path.join(split_dir, fn)
        try:
            # 加载.npz文件
            data = np.load(file_path)

            # 检查文件是否包含'ct'数组
            if 'ct' not in data:
                print(f"文件 {fn} 不包含 'ct' 数组")
                continue

            arr = data['ct']

            # 检查数组形状和维度
            print(f"文件 {fn} 的形状: {arr.shape}")

            # 检查是否有足够的列（至少4列）
            if arr.shape[1] < 4:
                print(f"文件 {fn} 只有 {arr.shape[1]} 列，需要至少4列")
                continue

            # 提取标签列（第4列，索引为3）
            labels = arr[:, 3]

            # 统计非零值
            nonzero_count = np.count_nonzero(labels)
            unique_labels = np.unique(labels)

            print(f"文件 {fn}: 总点数={len(labels)}, 非零点数={nonzero_count}, 唯一标签值={unique_labels}")

            file_count += 1
            if nonzero_count > 0:
                files_with_nonzero_labels += 1

        except Exception as e:
            print(f"处理文件 {fn} 时出错: {str(e)}")

    print(f"\n总结 - {split_dir}:")
    print(f"总共检查了 {file_count} 个文件")
    print(f"包含非零标签的文件: {files_with_nonzero_labels}")
    print(f"所有标签都为零的文件: {file_count - files_with_nonzero_labels}")


def main():
    parser = argparse.ArgumentParser(description='检查预处理数据集的标签分布')
    parser.add_argument('--root', default=r'./dataset/seg_input_10w', help='数据集根目录')
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.root, split)
        print(f"\n=== 检查 {split} 分割 ===")
        check_npz_files(split_dir)


if __name__ == '__main__':
    main()
