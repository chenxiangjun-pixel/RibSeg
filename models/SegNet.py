import torch
import torch.nn as nn
import torch.nn.functional as F
from .pn2 import PointNet2


class SegNet(nn.Module):
    """基于 PointNet++ 骨干的简易分割网络。"""

    def __init__(self, cls_num: int = 2):
        super().__init__()
        self.backbone = PointNet2()
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, cls_num, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数
        ------
        xyz: torch.Tensor
            输入点云，形状为 (B, 3, N)。

        返回
        ------
        torch.Tensor
            每个点的类别得分，形状为 (B, cls_num, N)。
        """
        x = self.backbone(xyz)
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        return x


class SegLoss(nn.Module):
    """用于点级分割的交叉熵损失。"""

    def __init__(self, cls_num: int = 2, weight: torch.Tensor | None = None):
        """参数
        ------
        cls_num: int
            类别数。
        weight: torch.Tensor | None
            传入 ``nn.CrossEntropyLoss`` 的权重，用于缓解类别不平衡。
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算分割损失。

        参数
        ------
        pred: torch.Tensor
            预测得分，形状为 (B, C, N)。
        target: torch.Tensor
            真实标签，形状为 (B, N)。
        """
        pred = pred.permute(0, 2, 1).contiguous()
        pred = pred.view(-1, pred.shape[-1])
        target = target.view(-1)
        return self.criterion(pred, target)
