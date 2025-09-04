import argparse
import os
from pathlib import Path
import importlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_utils.dataloader import RibSegDataset
import data_utils.data_aug as data_aug
from implement_class_balancing import compute_class_weights


def parse_args():
    parser = argparse.ArgumentParser('训练')
    parser.add_argument('--model', type=str, default='SegNet', help='模型名称')
    parser.add_argument('--batch_size', type=int, default=4, help='训练时的批量大小')
    parser.add_argument('--epoch', type=int, default=300, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='初始学习率')
    parser.add_argument('--gpu', type=str, default='0', help='使用的 GPU 编号')
    parser.add_argument('--optimizer', type=str, default='Adam', help='优化器: Adam 或 SGD')
    parser.add_argument('--log_dir', type=str, default='./log', help='保存检查点和日志的目录')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--root', type=str, default='./dataset/seg_input_10w', help='数据集根目录')
    parser.add_argument('--npoint', type=int, default=30000, help='采样点数')
    parser.add_argument('--step_size', type=int, default=20, help='学习率衰减步长')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='学习率衰减倍率')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'best_model.pth'

    # 数据集
    train_dataset = RibSegDataset(root=args.root, npoints=args.npoint, split='train', flag_arpe=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = RibSegDataset(root=args.root, npoints=args.npoint, split='test', flag_arpe=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"The number of labeled training data is: {len(train_dataset)}")
    print(f"The number of test data is: {len(test_dataset)}")

    # 模型
    MODEL = importlib.import_module('models.' + args.model)
    cls_num = 2
    classifier = MODEL.SegNet(cls_num=cls_num).cuda()
    # 计算类别权重以缓解不平衡
    _, w = compute_class_weights(args.root)
    weight = torch.from_numpy(w.astype(np.float32)).cuda()
    print('Using class weights:', w)
    criterion = MODEL.SegLoss(cls_num=cls_num, weight=weight).cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    best_acc = -1.0
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print('Use pretrain model, best_acc:', best_acc)
    else:
        print('No existing model, starting training from scratch...')
        classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    logs = []

    for epoch in range(start_epoch, args.epoch):
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        momentum = max(momentum, 0.01)
        classifier.apply(
            lambda x: x.__setattr__('momentum', momentum)
            if isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d)) else None
        )

        ep_loss = 0.0
        mean_correct = []
        for pc, label in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
            pc = pc.numpy()
            pc[:, :, :3] = data_aug.jitter_point_cloud(pc[:, :, :3], 0.005, 0.01)
            pc[:, :, :3] = data_aug.random_scale_point_cloud(pc[:, :, :3], 0.9, 1.1)
            pc = torch.from_numpy(pc).float().cuda()
            label = label.long().cuda()
            pc = pc.transpose(2, 1)
            if cls_num == 2:
                label[label != 0] = 1

            optimizer.zero_grad()
            classifier.train()
            pred = classifier(pc)
            seg_pred_choice = pred.contiguous().view(-1, cls_num)
            pred_choice = seg_pred_choice.data.max(1)[1]
            correct = pred_choice.eq(label.view(-1)).sum()
            mean_correct.append(correct.item() / label.numel())
            loss = criterion(pred, label)
            ep_loss += loss.item()
            loss.backward()
            optimizer.step()

        ep_loss /= len(train_loader)
        train_acc = float(np.mean(mean_correct))
        print(f'Train accuracy of seg is: {train_acc:.5f}')
        print(f'Train loss is: {ep_loss:.5f}')

        with torch.no_grad():
            mean_correct_test = []
            for pc, label in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
                pc, label = pc.float().cuda(), label.long().cuda()
                pc = pc.transpose(2, 1)
                if cls_num == 2:
                    label[label != 0] = 1
                classifier.eval()
                pred = classifier(pc)
                seg_pred_choice = pred.contiguous().view(-1, cls_num)
                pred_choice = seg_pred_choice.data.max(1)[1]
                correct = pred_choice.eq(label.view(-1)).sum()
                mean_correct_test.append(correct.item() / label.numel())
            test_acc = float(np.mean(mean_correct_test))
            print(f'Test accuracy of seg is: {test_acc:.5f}')

        logs.append({'epoch': epoch, 'lr': lr, 'train_loss': ep_loss, 'train_acc': train_acc, 'test_acc': test_acc})

        if test_acc >= best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, ckpt_path)
            print(f'Saving best model at epoch {epoch} with acc {best_acc:.5f}')

    pd.DataFrame(logs).to_excel(save_dir / 'train_log.xlsx', index=False)
    print(f'Training log saved to {save_dir / "train_log.xlsx"}')


if __name__ == '__main__':
    main(parse_args())
