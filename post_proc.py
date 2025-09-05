import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
# from npz_to_mask import npz_to_mask
import SimpleITK as sitk

def ensure_coord_order(pts: np.ndarray, shape):
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


def extract_centerline(mask: np.ndarray) -> np.ndarray:
    """Extract a 1-voxel-wide centerline skeleton from a binary mask."""
    itk_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    skeleton = sitk.BinaryThinning(itk_img)
    return sitk.GetArrayFromImage(skeleton)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)


def process_case(npz_file: Path, ct_dir: str, label_dir: str, out_dir: str, min_size: int):
    case = npz_file.stem
    ct_path = Path(ct_dir) / f"{case}-image.nii.gz"
    out_path = Path(out_dir) / f"{case}_mask.nii"
    npz_to_mask(str(npz_file), str(ct_path), str(out_path), min_size)

    mask_img = nib.load(str(out_path))
    mask = mask_img.get_fdata()

    centerline = extract_centerline(mask)
    centerline_path = Path(out_dir) / f"{case}_centerline.nii"
    nib.save(nib.Nifti1Image(centerline.astype(np.uint8), mask_img.affine), centerline_path)

    dice = None
    if label_dir:
        label_candidates = [Path(label_dir) / f"{case}-label.nii.gz"]
        label_path = next((p for p in label_candidates if p.exists()), None)
        if label_path is not None:
            gt = nib.load(str(label_path)).get_fdata()
            dice = dice_score(mask, gt)
    return case, dice


def main():
    parser = argparse.ArgumentParser(description="Convert RibSeg npz predictions to NIfTI masks and optionally compute Dice scores")
    parser.add_argument('--npz_dir', default=r"E:\Code_collection\8_Rib_seg\RibSeg\result", help='Directory containing RibFracxxx.npz files')
    parser.add_argument('--ct_dir', default=r'F:\Data_Collection\Ribfrac\Part2\image', help='Directory with original CT volumes')
    parser.add_argument('--out_dir', default=r'E:\Code_collection\8_Rib_seg\RibSeg\result\结果可视化nii', help='Directory to save NIfTI masks')
    parser.add_argument('--label_dir', default=r'F:\Data_Collection\Ribfrac\Part2\label', help='Directory with ground-truth labels for evaluation')
    parser.add_argument('--min_size', type=int, default=1000, help='Minimum size to retain connected components')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    dices = []
    for npz_file in sorted(Path(args.npz_dir).glob('*.npz')):
        case, dice = process_case(npz_file, args.ct_dir, args.label_dir, args.out_dir, args.min_size)
        if dice is not None:
            dices.append(dice)
            print(f"{case}: Dice={dice:.4f}")
        else:
            print(f"{case}: saved mask")

    if dices:
        print(f"Average Dice: {np.mean(dices):.4f}")


if __name__ == '__main__':
    main()
