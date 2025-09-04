import torch
import torch.nn as nn
import torch.nn.functional as F
from .pn2_util import PointNetSetAbstractionMsg, PointNetFeaturePropagation



class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])

        self.fp4 = PointNetFeaturePropagation(512+1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(96+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self, xyz):
        """
          输入:
              xyz: 点的位置数据，形状为 [B, 3, N]
          返回:
              feat: 特征，形状为 [B, 128, N]
        """
        # 集成抽象层
        l0_points = xyz  # [B, 3, 30000]
        l0_xyz = xyz[:,:3,:]  # [B, 3, 30000]


        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B,3,1024], [B,96,1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,3,256], [B,256,256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,3,64], [B,512,64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [B,3,16], [B,1024,16]
  
        # print(l3_xyz.shape,l3_points.shape,)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [B, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)     # [B, 128, 30000]
 
        # print(l0_points.shape)
        # return l3_xyz,l2_xyz,l1_xyz,l0_xyz,l3_points,l2_points,l1_points,l0_points
        return l0_points

