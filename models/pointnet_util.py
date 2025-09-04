import torch
import torch.nn as nn
import torch.nn.functional as F


def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """计算两组点之间的平方距离。

    参数
    ------
    src: torch.Tensor
        源点集，形状为 (B, N, C)
    dst: torch.Tensor
        目标点集，形状为 (B, M, C)

    返回
    ------
    torch.Tensor
        距离矩阵，形状为 (B, N, M)
    """
    return torch.cdist(src, dst, p=2) ** 2


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """根据索引张量提取点。"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (idx.dim() - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """最远点采样。"""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def _knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    dist = _square_distance(new_xyz, xyz)
    _, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=False)
    return idx


class PointNetSetAbstractionMsg(nn.Module):
    """多尺度分组的集合抽象层。"""

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlps = nn.ModuleList()
        for mlp in mlp_list:
            layers = []
            # 每个局部区域将归一化的 xyz（3 维）与已有点特征拼接，
            # 因此第一层卷积的输入通道数为 ``in_channel + 3``。
            last_channel = in_channel + 3
            for out_channel in mlp:
                layers.append(nn.Conv2d(last_channel, out_channel, 1))
                layers.append(nn.BatchNorm2d(out_channel))
                layers.append(nn.ReLU())
                last_channel = out_channel
            self.mlps.append(nn.Sequential(*layers))

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        B, _, N = xyz.shape
        xyz_t = xyz.transpose(1, 2)
        if self.npoint is None:
            new_xyz = xyz_t.mean(dim=1, keepdim=True)
        else:
            fps_idx = _farthest_point_sample(xyz_t, self.npoint)
            new_xyz = _index_points(xyz_t, fps_idx)
        new_points_list = []
        for i, mlp in enumerate(self.mlps):
            k = self.nsample_list[i]
            group_idx = _knn_point(k, xyz_t, new_xyz)
            grouped_xyz = _index_points(xyz_t, group_idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
            grouped_xyz_norm = grouped_xyz_norm.permute(0, 3, 1, 2)
            if points is not None:
                grouped_points = _index_points(points.transpose(1, 2), group_idx)
                grouped_points = grouped_points.permute(0, 3, 1, 2)
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)
            else:
                grouped_points = grouped_xyz_norm
            new_points = mlp(grouped_points)
            new_points = torch.max(new_points, -1)[0]
            new_points_list.append(new_points)
        new_points = torch.cat(new_points_list, dim=1)
        return new_xyz.transpose(1, 2).contiguous(), new_points


class PointNetSetAbstraction(nn.Module):
    """单尺度分组的集合抽象层。"""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all
        layers = []
        # 与 MSG 抽象类似，每个局部组将 xyz 偏移（3 通道）与输入特征拼接。
        last_channel = in_channel + 3
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        B, _, N = xyz.shape
        xyz_t = xyz.transpose(1, 2)
        if self.group_all:
            new_xyz = xyz_t.mean(dim=1, keepdim=True)
            grouped_xyz = xyz_t.unsqueeze(1)
            if points is not None:
                grouped_points = points.transpose(1, 2).unsqueeze(1)
                grouped_points = torch.cat([grouped_xyz - new_xyz.unsqueeze(2), grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz - new_xyz.unsqueeze(2)
        else:
            fps_idx = _farthest_point_sample(xyz_t, self.npoint)
            new_xyz = _index_points(xyz_t, fps_idx)
            group_idx = _knn_point(self.nsample, xyz_t, new_xyz)
            grouped_xyz = _index_points(xyz_t, group_idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = _index_points(points.transpose(1, 2), group_idx)
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz_norm
            grouped_points = grouped_points.permute(0, 3, 1, 2)
        new_points = self.mlp(grouped_points.permute(0,3,1,2)) if self.group_all else self.mlp(grouped_points)
        new_points = torch.max(new_points, -1)[0]
        return new_xyz.transpose(1,2).contiguous(), new_points


class PointNetFeaturePropagation(nn.Module):
    """用于上采样的特征传播模块。"""

    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        B, _, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_t, xyz2_t = xyz1.transpose(1,2), xyz2.transpose(1,2)
        if N2 == 1:
            interpolated_points = points2.expand(-1, -1, N1)
        else:
            k = min(3, N2)
            idx = _knn_point(k, xyz2_t, xyz1_t)
            grouped_xyz = _index_points(xyz2_t, idx)  # 形状为 (B, N1, k, 3)
            dist = torch.sum((xyz1_t.unsqueeze(2) - grouped_xyz) ** 2, dim=-1)
            dist = torch.clamp(dist, min=1e-10)
            weight = 1.0 / dist
            weight = weight / torch.sum(weight, dim=-1, keepdim=True)
            grouped_points = _index_points(points2.transpose(1, 2), idx)
            interpolated_points = torch.sum(grouped_points * weight.unsqueeze(-1), dim=2)
            interpolated_points = interpolated_points.permute(0, 2, 1)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
        new_points = self.mlp(new_points)
        return new_points
