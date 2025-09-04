"""使用实际点数计算每个批次的准确率。

旧版训练循环将正确预测数除以固定的 ``batch_size * npoint``，在最后一
个不完整批次上会低估准确率。该示例演示如何直接根据张量计算分母，确保
无论批次大小如何都能得到正确的指标。

用法::
    python use_true_batch_size_for_accuracy.py
"""

import torch


def batch_accuracy(pred: torch.Tensor, label: torch.Tensor) -> float:
    """计算一个批次的准确率。

    参数
    ------
    pred : torch.Tensor
        形状为 ``(B, C, N)`` 的类别分数张量。
    label : torch.Tensor
        形状为 ``(B, N)`` 的真实标签张量。
    """
    pred_choice = pred.max(dim=1)[1]
    total_points = label.numel()
    correct = pred_choice.eq(label).sum().item()
    return correct / total_points


def main():
    pred = torch.randn(2, 2, 30000)
    label = torch.randint(0, 2, (2, 30000))
    acc = batch_accuracy(pred, label)
    print('Accuracy:', acc)


if __name__ == '__main__':
    main()
