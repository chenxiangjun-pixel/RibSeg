from __future__ import annotations
import argparse
import os
import nibabel as nib
import numpy as np


def process_part2(root_dir: str) -> None:
    """Convert RibFrac Part2 images and labels into point-cloud numpy arrays.

    Parameters
    ----------
    root_dir : str
        Path containing ``image`` and ``label`` subdirectories.
    """

    img_dir = os.path.join(root_dir, "image")
    label_dir = os.path.join(root_dir, "label")

    if not os.path.isdir(img_dir):
        print(f"{img_dir} not found, skip")
        return

    os.makedirs("../data/pn/data_pn", exist_ok=True)
    os.makedirs("../data/pn/label_pn", exist_ok=True)

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
            print(f"Label for {img_name} not found, skip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RibFrac Part2 data")
    parser.add_argument("--root", default=r"E:\Data_collection\Ribfrac\Part2",
        help="Path to Part2 directory containing 'image' and 'label' folders")
    args = parser.parse_args()

    process_part2(args.root)


if __name__ == "__main__":
    main()
