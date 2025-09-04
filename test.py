import argparse
import os
import importlib
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from data_utils.dataloader import RibSegDataset


def parse_args():
    parser = argparse.ArgumentParser('测试')
    parser.add_argument('--model', type=str, default='SegNet', help='模型名称')
    parser.add_argument('--batch_size', type=int, default=4, help='测试时的批量大小')
    parser.add_argument('--gpu', type=str, default='0', help='使用的 GPU 编号')
    parser.add_argument('--log_dir', type=str, required=True, help='包含 best_model.pth 的目录')
    parser.add_argument('--root', type=str, default='./dataset/seg_input_10w', help='数据集根目录')
    parser.add_argument('--npoint', type=int, default=30000, help='采样点数')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir = Path(args.log_dir)
    ckpt_path = save_dir / 'best_model.pth'

    test_dataset = RibSegDataset(root=args.root, npoints=args.npoint, split='test', flag_arpe=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    MODEL = importlib.import_module('models.' + args.model)
    cls_num = 2
    classifier = MODEL.SegNet(cls_num=cls_num).cuda()

    checkpoint = torch.load(str(ckpt_path), weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    tp = tn = fp = fn = 0
    with torch.no_grad():
        for pc, label in tqdm(test_loader, total=len(test_loader)):
            pc, label = pc.float().cuda(), label.long().cuda()
            pc = pc.transpose(2, 1)
            if cls_num == 2:
                label[label != 0] = 1
            pred = classifier(pc)
            seg_pred_choice = pred.contiguous().view(-1, cls_num)
            pred_choice = seg_pred_choice.data.max(1)[1]
            label_flat = label.view(-1)
            tp += ((pred_choice == 1) & (label_flat == 1)).sum().item()
            tn += ((pred_choice == 0) & (label_flat == 0)).sum().item()
            fp += ((pred_choice == 1) & (label_flat == 0)).sum().item()
            fn += ((pred_choice == 0) & (label_flat == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    metrics = {
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
    }
    for k, v in metrics.items():
        print(f'{k}: {v:.5f}')

    pd.DataFrame([metrics]).to_excel(save_dir / 'test_metrics.xlsx', index=False)
    print(f'Test metrics saved to {save_dir / "test_metrics.xlsx"}')


if __name__ == '__main__':
    main(parse_args())
