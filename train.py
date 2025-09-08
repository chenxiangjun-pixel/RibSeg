import argparse
import os
from pathlib import Path
import importlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.dataloader import RibSegDataset
import data_utils.data_aug as data_aug
from utils.implement_class_balancing import compute_class_weights
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser('训练')
    parser.add_argument('--model', type=str, default='SegNet', help='模型名称')

    parser.add_argument('--batch_size', type=int, default=32, help='训练时的批量大小')
    parser.add_argument('--epoch', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='初始学习率')
    parser.add_argument('--gpu', type=str, default='0', help='使用的 GPU 编号')
    parser.add_argument('--optimizer', type=str, default='Adam', help='优化器: Adam 或 SGD')
    parser.add_argument('--log_dir', type=str, default='./log', help='保存检查点和日志的目录')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--root', type=str, default='./dataset/seg_input_10w', help='数据集根目录')
    parser.add_argument('--npoint', type=int, default=30000, help='采样点数')
    parser.add_argument('--step_size', type=int, default=20, help='学习率衰减步长')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='学习率衰减倍率')
    parser.add_argument('--use_cpu', action='store_true', help='使用CPU训练')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载 workers 数量')
    parser.add_argument('--loss_fn', type=str, default='focal',
                        choices=['ce', 'dice', 'combined', 'hard_dice', 'combined_hard', 'focal'],
                        help='损失函数: ce(交叉熵), dice(Dice损失), combined(组合损失), hard_dice(硬标签Dice), combined_hard(硬标签组合), focal(Focal Loss)')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'plateau', 'cosine'],
                        help='学习率调度器: step(步长衰减), plateau(平台衰减), cosine(余弦退火)')
    return parser.parse_args()


# 硬标签Dice损失函数
class HardDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(HardDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 获取硬预测
        preds = inputs.argmax(dim=1)

        # 转换为one-hot
        preds_onehot = F.one_hot(preds, num_classes=inputs.shape[1]).float().permute(0, 2, 1)
        targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1]).float().permute(0, 2, 1)

        # 计算Dice
        intersection = (preds_onehot * targets_onehot).sum()
        union = preds_onehot.sum() + targets_onehot.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


# 硬标签组合损失函数（硬Dice + 交叉熵）
class CombinedHardLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5):
        super(CombinedHardLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = HardDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * ce_loss


# Focal Loss处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 稀疏性正则化损失
class SparsityRegularizedLoss(nn.Module):
    def __init__(self, base_loss, sparsity_weight=0.5):
        super(SparsityRegularizedLoss, self).__init__()
        self.base_loss = base_loss
        self.sparsity_weight = sparsity_weight

    def forward(self, inputs, targets):
        base_loss = self.base_loss(inputs, targets)

        # 计算预测为正类的比例，鼓励稀疏性
        pred_probs = F.softmax(inputs, dim=1)
        positive_ratio = pred_probs[:, 1, :].mean()  # 假设类别1是正类
        sparsity_loss = positive_ratio  # 我们希望这个值小

        return base_loss + self.sparsity_weight * sparsity_loss


# 计算分割指标
def calculate_segmentation_metrics(pred, target, smooth=1e-6):
    """
    使用argmax后的硬标签计算指标，确保与可视化逻辑一致。
    pred: 网络的原始输出logits [B, C, N]
    target: 真实标签 [B, N]
    """
    # 获取预测的硬标签
    pred_binary = pred.argmax(dim=1)  # [B, N]

    # 将预测和真实标签都转换为one-hot，用于计算Dice/IoU
    num_classes = pred.shape[1]
    pred_onehot = F.one_hot(pred_binary, num_classes=num_classes).float().permute(0, 2, 1)  # [B, C, N]
    target_onehot = F.one_hot(target, num_classes=num_classes).float().permute(0, 2, 1)  # [B, C, N]

    # 计算交集和并集 (只对类别1，即肋骨计算)
    # 假设肋骨是类别1
    intersection = (pred_onehot[:, 1, :] * target_onehot[:, 1, :]).sum()
    union = pred_onehot[:, 1, :].sum() + target_onehot[:, 1, :].sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)

    # 计算真正例、假正例、真负例、假负例
    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()

    sensitivity = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    precision = tp / (tp + fp + smooth)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + smooth)

    return {'dice': dice.item(),
            'iou': iou.item(),
            'sensitivity': sensitivity.item(),
            'specificity': specificity.item(),
            'precision': precision.item(),
            'f1': f1.item()}


def train_epoch(classifier, train_loader, optimizer, criterion, cls_num, device):
    classifier.train()
    ep_loss = 0.0
    metrics_accumulator = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, 'f1': 0.0}

    for i, (pc, label) in enumerate(tqdm(train_loader, total=len(train_loader), smoothing=0.9)):
        # 数据增强在CPU上进行
        pc_np = pc.numpy()
        pc_np[:, :, :3] = data_aug.jitter_point_cloud(pc_np[:, :, :3], 0.005, 0.01)
        pc_np[:, :, :3] = data_aug.random_scale_point_cloud(pc_np[:, :, :3], 0.9, 1.1)
        pc = torch.from_numpy(pc_np).float()
        # 移动到设备
        pc, label = pc.to(device), label.to(device)
        # 确保标签是长整型
        label = label.long()
        # 检查数据形状
        if pc.shape[0] == 0 or pc.shape[1] == 0 or pc.shape[2] == 0:
            print(f"Invalid data shape: {pc.shape}, skipping batch")
            continue

        pc = pc.transpose(2, 1)
        if cls_num == 2:
            label[label != 0] = 1
        optimizer.zero_grad()
        try:
            pred = classifier(pc)
            # 计算分割指标
            metrics = calculate_segmentation_metrics(pred, label)
            for key in metrics_accumulator:
                metrics_accumulator[key] += metrics[key]
            # 计算损失
            loss = criterion(pred, label)
            ep_loss += loss.item()
            loss.backward()
            # 确保优化器状态在CPU上
            if torch.cuda.is_available():
                for param in optimizer.param_groups[0]['params']:
                    state = optimizer.state[param]
                    if 'step' in state:
                        state['step'] = state['step'].cpu()

            optimizer.step()
        except RuntimeError as e:
            print(f"Runtime error during training: {e}")
            # 清除GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Other error during training: {e}")
            continue

    # 计算平均指标
    num_batches = len(train_loader)
    ep_loss /= num_batches if num_batches > 0 else 1.0
    for key in metrics_accumulator:
        metrics_accumulator[key] /= num_batches if num_batches > 0 else 1.0

    return ep_loss, metrics_accumulator


def test_epoch(classifier, test_loader, cls_num, device):
    classifier.eval()
    metrics_accumulator = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, 'f1': 0.0}

    with torch.no_grad():
        for pc, label in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
            pc, label = pc.to(device), label.to(device)
            # 确保标签是长整型
            label = label.long()
            # 检查数据形状
            if pc.shape[0] == 0 or pc.shape[1] == 0 or pc.shape[2] == 0:
                print(f"Invalid data shape: {pc.shape}, skipping batch")
                continue
            pc = pc.transpose(2, 1)
            if cls_num == 2:
                if cls_num == 2:
                    label[label != 0] = 1
            try:
                pred = classifier(pc)
                # 计算分割指标
                metrics = calculate_segmentation_metrics(pred, label)
                for key in metrics_accumulator:
                    metrics_accumulator[key] += metrics[key]
            except RuntimeError as e:
                print(f"Runtime error during testing: {e}")
                # 清除GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Other error during testing: {e}")
                continue

    # 计算平均指标
    num_batches = len(test_loader)
    for key in metrics_accumulator:
        metrics_accumulator[key] /= num_batches if num_batches > 0 else 1.0

    return metrics_accumulator


def main(args):
    # 设备设置
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    save_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'best_model.pth'
    # 预计算类别权重
    weights_path = save_dir / 'class_weights.npy'
    if weights_path.exists():
        w = np.load(weights_path)
    else:
        _, w = compute_class_weights(args.root)
        np.save(weights_path, w)

    weight = torch.from_numpy(w.astype(np.float32)).to(device)
    print('Using class weights:', w)

    # 数据集 - 设置num_workers=0避免多进程问题
    train_dataset = RibSegDataset(root=args.root, npoints=args.npoint, split='train', flag_arpe=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=device.type != 'cpu')
    test_dataset = RibSegDataset(root=args.root, npoints=args.npoint, split='test', flag_arpe=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=device.type != 'cpu')
    print(f"The number of labeled training data is: {len(train_dataset)}")
    print(f"The number of test data is: {len(test_dataset)}")

    # 初始化TensorBoard SummaryWriter
    tb_writer = SummaryWriter(log_dir=save_dir / 'tensorboard')

    # 模型
    MODEL = importlib.import_module('models.' + args.model)
    cls_num = 2
    classifier = MODEL.SegNet(cls_num=cls_num).to(device)

    # 选择损失函数
    if args.loss_fn == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        print("Using CrossEntropy loss")
    elif args.loss_fn == 'dice':
        # 使用原有的软标签Dice损失
        class DiceLoss(nn.Module):
            def __init__(self, smooth=1.0):
                super(DiceLoss, self).__init__()
                self.smooth = smooth

            def forward(self, inputs, targets):
                inputs = F.softmax(inputs, dim=1)
                targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
                targets_onehot = targets_onehot.permute(0, 2, 1)
                intersection = (inputs * targets_onehot).sum()
                union = inputs.sum() + targets_onehot.sum()
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                return 1 - dice

        criterion = DiceLoss().to(device)
        print("Using Dice loss")
    elif args.loss_fn == 'combined':
        # 使用原有的软标签组合损失
        class DiceLoss(nn.Module):
            def __init__(self, smooth=1.0):
                super(DiceLoss, self).__init__()
                self.smooth = smooth

            def forward(self, inputs, targets):
                inputs = F.softmax(inputs, dim=1)
                targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
                targets_onehot = targets_onehot.permute(0, 2, 1)
                intersection = (inputs * targets_onehot).sum()
                union = inputs.sum() + targets_onehot.sum()
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                return 1 - dice

        class CombinedLoss(nn.Module):
            def __init__(self, weight=None, alpha=0.5):
                super(CombinedLoss, self).__init__()
                self.alpha = alpha
                self.dice_loss = DiceLoss()
                self.ce_loss = nn.CrossEntropyLoss(weight=weight)

            def forward(self, inputs, targets):
                dice_loss = self.dice_loss(inputs, targets)
                ce_loss = self.ce_loss(inputs, targets)
                return self.alpha * dice_loss + (1 - self.alpha) * ce_loss

        criterion = CombinedLoss(weight=weight).to(device)
        print("Using Combined (Dice + CrossEntropy) loss")
    elif args.loss_fn == 'hard_dice':
        criterion = HardDiceLoss().to(device)
        print("Using Hard Dice loss")
    elif args.loss_fn == 'combined_hard':
        criterion = CombinedHardLoss(weight=weight).to(device)
        print("Using Combined Hard (Hard Dice + CrossEntropy) loss")
    elif args.loss_fn == 'focal':
        criterion = FocalLoss().to(device)
        print("Using Focal loss")

    # 添加稀疏性正则化
    if args.loss_fn in ['hard_dice', 'combined_hard', 'focal']:
        criterion = SparsityRegularizedLoss(criterion, sparsity_weight=0.1)
        print("Added sparsity regularization to loss function")

    # 检查数据集中的标签类型
    sample_pc, sample_label = next(iter(train_loader))
    print(f"Sample label type: {sample_label.dtype}")
    print(f"Sample label shape: {sample_label.shape}")
    print(f"Sample point cloud shape: {sample_pc.shape}")

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    # 先定义优化器
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    # 选择学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
        print("Using StepLR scheduler")
    elif args.lr_scheduler == 'plateau':
        # 对于ReduceLROnPlateau，确保所有状态都在CPU上
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                state = optimizer.state[param]
                if 'step' in state:
                    state['step'] = state['step'].cpu()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        print("Using ReduceLROnPlateau scheduler")
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
        print("Using CosineAnnealingLR scheduler")

    start_epoch = 0
    best_dice = -1.0

    # 检查点加载
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])

        # 只有在检查点中有优化器状态时才加载
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Optimizer state loaded')

            # 确保优化器状态在CPU上
            if torch.cuda.is_available():
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        state = optimizer.state[param]
                        if 'step' in state:
                            state['step'] = state['step'].cpu()

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('Scheduler state loaded')

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice = checkpoint.get('best_dice', -1.0)
        print(f'Use pretrain model, starting from epoch {start_epoch}, best_dice: {best_dice}')
    else:
        print('No existing model, starting training from scratch...')
        classifier.apply(weights_init)

    logs = []

    for epoch in range(start_epoch, args.epoch):
        print(f'Epoch {epoch}/{args.epoch}:')
        # 训练阶段
        train_loss, train_metrics = train_epoch(classifier, train_loader, optimizer, criterion, cls_num, device)
        # 测试阶段
        test_metrics = test_epoch(classifier, test_loader, cls_num, device)
        # 更新学习率
        if args.lr_scheduler == 'plateau':
            # 确保损失值在CPU上
            scheduler.step(float(train_loss))
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # 记录日志
        log_entry = {'epoch': epoch, 'lr': current_lr, 'train_loss': train_loss}
        for key in train_metrics:
            log_entry[f'train_{key}'] = train_metrics[key]
        for key in test_metrics:
            log_entry[f'test_{key}'] = test_metrics[key]
        logs.append(log_entry)

        # 记录到TensorBoard
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Learning_rate', current_lr, epoch)

        # 记录训练指标
        for key, value in train_metrics.items():
            tb_writer.add_scalar(f'Train/{key}', value, epoch)

        # 记录测试指标
        for key, value in test_metrics.items():
            tb_writer.add_scalar(f'Test/{key}', value, epoch)

        # 打印训练和测试指标
        print(
            f'Train - Loss: {train_loss:.3f} || Dice: {train_metrics["dice"]:.3f} || IoU: {train_metrics["iou"]:.3f} || Sensitivity: {train_metrics["sensitivity"]:.3f} '
            f'|| Specificity: {train_metrics["specificity"]:.3f} || Precision: {train_metrics["precision"]:.3f} || F1: {train_metrics["f1"]:.3f}')

        print(
            f'Test - Dice: {test_metrics["dice"]:.3f} || IoU: {test_metrics["iou"]:.3f} || Sensitivity: {test_metrics["sensitivity"]:.5f} || '
            f'Specificity: {test_metrics["specificity"]:.3f} || 'f'Precision: {test_metrics["precision"]:.3f} || F1: {test_metrics["f1"]:.3f}')

        print(f'Learning rate: {current_lr:.3f}')
        print("--" * 50)

        # 保存最佳模型（基于Dice系数）
        if test_metrics["dice"] >= best_dice:
            best_dice = test_metrics["dice"]
            state = {'epoch': epoch, 'best_dice': best_dice, 'model_state_dict': classifier.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
            torch.save(state, ckpt_path)
            print(f'Saving best model at epoch {epoch} with dice {best_dice:.3f}')

        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(state, checkpoint_path)

        # 清除GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存最终模型
    torch.save(classifier.state_dict(), save_dir / 'final_model.pth')

    # 保存日志
    pd.DataFrame(logs).to_excel(save_dir / 'train_log.xlsx', index=False)
    print(f'Training log saved to {save_dir / "train_log.xlsx"}')

    # 关闭TensorBoard writer
    tb_writer.close()
    print(f'TensorBoard logs saved to {save_dir / "tensorboard"}')


if __name__ == '__main__':
    args = parse_args()
    # 设置环境变量以捕获CUDA错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main(args)