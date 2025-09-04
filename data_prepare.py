"""从 RibFrac 第二部分体积数据生成点云训练数据。

脚本将 RibFrac 数据集中成对的 CT 体数据与肋骨分割标签转换为稀疏点云。
其适用于文件组织形式为::

    <root>/image/RibFrac301-image.nii.gz
    <root>/label/RibFrac301-label.nii.gz

HU 值大于等于 200 的体素坐标将保存到 ``./data/pn/data_pn``，对应的标签
保存到 ``./data/pn/label_pn``，输出文件名与病例编号一致（如示例中的
``RibFrac301``）。
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np


def process_part2(root_dir: str) -> None:
    """将 RibFrac 第二部分的图像与标签转为点云数组。

    参数
    ------
    root_dir : str
        包含 ``image`` 与 ``label`` 子目录的路径。
    """

    img_dir = os.path.join(root_dir, "image")
    label_dir = os.path.join(root_dir, "label")

    if not os.path.isdir(img_dir):
        print(f"{img_dir} 不存在，跳过")
        return

    os.makedirs("./data/pn/data_pn", exist_ok=True)
    os.makedirs("./data/pn/label_pn", exist_ok=True)

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.endswith("-image.nii.gz"):
            continue

        img_path = os.path.join(img_dir, img_name)
        label_name = img_name.replace("-image", "-label")
        label_path = os.path.join(label_dir, label_name)

        try:
            source = nib.load(img_path).get_fdata()
            source = (source >= 200).astype(np.uint8)
            label = nib.load(label_path).get_fdata()

            coords = np.argwhere(source == 1)
            labels = label[coords[:, 0], coords[:, 1], coords[:, 2]]

            case_id = img_name.replace("-image.nii.gz", "")
            np.save(f"./data/pn/data_pn/{case_id}", coords)
            np.save(f"./data/pn/label_pn/{case_id}", labels)
        except FileNotFoundError:
            print(f"{img_name} 缺少标签文件，跳过")


def main() -> None:
    parser = argparse.ArgumentParser(description="处理 RibFrac Part2 数据")
    parser.add_argument(
        "--root",
        default="./data/ribfrac/Part2",
        help="包含 'image' 与 'label' 文件夹的 Part2 目录路径",
    )
    args = parser.parse_args()

    process_part2(args.root)


if __name__ == "__main__":
    main()

