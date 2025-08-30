import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pn2 import PointNet2


class SegNet(nn.Module):
    """Simple segmentation network built on PointNet++ backbone."""

    def __init__(self, cls_num: int = 2):
        super().__init__()
        self.backbone = PointNet2()
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, cls_num, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        xyz: torch.Tensor
            Input point cloud of shape (B, 3, N).

        Returns
        -------
        torch.Tensor
            Per-point class scores with shape (B, cls_num, N).
        """
        x = self.backbone(xyz)
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        return x


class SegLoss(nn.Module):
    """Cross‑entropy loss for point-wise segmentation."""

    def __init__(self, cls_num: int = 2):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute segmentation loss.

        Parameters
        ----------
        pred: torch.Tensor
            Predicted scores (B, C, N).
        target: torch.Tensor
            Ground-truth labels (B, N).
        """
        pred = pred.permute(0, 2, 1).contiguous()
        pred = pred.view(-1, pred.shape[-1])
        target = target.view(-1)
        return self.criterion(pred, target)