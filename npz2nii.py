"""直接看inference生成的result下面的npz结果"""
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os
import glob


def ensure_coord_order(pts, shape):
    """Ensure point coordinates match volume order (x, y, z).
    If coordinates exceed bounds, try swapping axes."""
    if np.all(pts[:, 0] < shape[0]) and np.all(pts[:, 1] < shape[1]) and np.all(pts[:, 2] < shape[2]):
        return pts
    if np.all(pts[:, 2] < shape[0]) and np.all(pts[:, 1] < shape[1]) and np.all(pts[:, 0] < shape[2]):
        return pts[:, [2, 1, 0]]
    raise ValueError('Coordinate indices do not fit volume shape; unable to determine (x, y, z) order.')


def npz_to_mask(npz_path: str, ct_path: str, out_path: str, min_size: int = 1000):
    data = np.load(npz_path)
    pts = data['ct'][:, :3].astype(np.int32)
    preds = data['seg'].astype(np.uint8)

    ct_img = nib.load(ct_path)
    mask = np.zeros(ct_img.shape, dtype=np.uint8)

    pts = ensure_coord_order(pts, mask.shape)
    mask[pts[:, 0], pts[:, 1], pts[:, 2]] = preds

    itk_mask = sitk.GetImageFromArray(mask)
    dilated = sitk.BinaryDilate(itk_mask, (1, 1, 1), sitk.sitkBall)

    cc = sitk.ConnectedComponent(dilated)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    keep = np.zeros(mask.shape, dtype=np.uint8)
    for label in stats.GetLabels():
        if stats.GetNumberOfPixels(label) >= min_size:
            keep |= sitk.GetArrayFromImage(cc == label)

    nib.save(nib.Nifti1Image(keep.astype(np.uint8), ct_img.affine), out_path)


def main():
    parser = argparse.ArgumentParser(description='Convert RibSeg npz results to NIfTI mask.')
    parser.add_argument('--npz_dir', default=r"E:\Code_collection\8_Rib_seg\RibSeg\result", help='Directory containing RibFracxxx.npz result files')
    parser.add_argument('--ct_dir', default=r'F:\Data_Collection\Ribfrac\Part2\image', help='Directory containing original CT NIfTI files')
    parser.add_argument('--out_dir', default=r'E:\Code_collection\8_Rib_seg\RibSeg\result\结果可视化nii', help='Output directory for NIfTI masks')
    parser.add_argument('--min_size', type=int, default=1000, help='Minimum size to keep connected components')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)

    # 获取所有NPZ文件
    npz_files = glob.glob(os.path.join(args.npz_dir, "RibFrac*.npz"))

    for npz_path in npz_files:
        # 从文件名提取ID
        base_name = os.path.basename(npz_path)
        file_id = base_name.split('.')[0]  # 例如 "RibFrac301"

        # 构建对应的CT文件路径
        ct_filename = f"{file_id}-image.nii.gz"
        ct_path = os.path.join(args.ct_dir, ct_filename)

        # 构建输出文件路径
        out_filename = f"{file_id}_mask.nii"
        out_path = os.path.join(args.out_dir, out_filename)

        # 检查CT文件是否存在
        if not os.path.exists(ct_path):
            print(f"Warning: CT file {ct_path} not found, skipping {npz_path}")
            continue

        print(f"Processing {npz_path} -> {out_path}")

        try:
            npz_to_mask(npz_path, ct_path, out_path, args.min_size)
            print(f"Successfully processed {file_id}")
        except Exception as e:
            print(f"Error processing {file_id}: {str(e)}")


if __name__ == '__main__':
    main()
