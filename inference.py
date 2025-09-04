import argparse
import os
from pickle import FALSE
import torch.nn.functional as F
import torch
from data_utils.dataloader import ARPD
import datetime
import logging
import time
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import nibabel as nib

# 如需调试 CUDA，可取消注释以下行：
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def pc_normalize(pc, centroid=None, m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    if m is None:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

    pc = pc - centroid
    pc = pc / m
    return pc, centroid, m

def parse_args():
    parser = argparse.ArgumentParser('模型参数')
    parser.add_argument('--model', type=str, default='CLNet', help='模型名称')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--gpu', type=str, default='0', help='使用的 GPU 编号')
    parser.add_argument('--log_dir', type=str, default=None, help='日志与模型目录')
    return parser.parse_args()


def main(args):
    """主函数：加载模型并执行推理。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/' + args.log_dir

    # 模型加载
    cls_num = 2

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.SegNet(cls_num=cls_num).cuda()

    checkpoint = torch.load(
        str(experiment_dir) + '/checkpoints/best_model.pth',
        weights_only=False,
    )
    classifier.load_state_dict(checkpoint['model_state_dict'],strict=True)
    arp = False

    data_dir = './dataset/ribseg/test/'

    dir_save = './res/'+args.log_dir+'/'
    if not os.path.exists(dir_save):
      os.makedirs(dir_save,exist_ok=True)

    data_list = [x for x in os.listdir(data_dir)]
    classifier = classifier.eval()
    with torch.no_grad():
        for name in tqdm(data_list):
            data = np.load(data_dir+name)['ct'].astype(np.float32)
            np.random.shuffle(data)
            ct_cords = data.copy()
            data[:,:3], centroid, m = pc_normalize(data[:,:3])
            ct_source_cord,seg = data[:,:3],data[:,3]
            
            sample_num = 30000*4
            num_p = ct_source_cord.shape[0]//sample_num
            data_to_do = []
            for i in range(num_p):
                data_to_do.append(ct_source_cord[sample_num*i:sample_num*(i+1),:])
            data_to_do.append(ct_source_cord[-sample_num:,:])

            pred = np.zeros(ct_source_cord.shape[0])

            for index in range(len(data_to_do)):
                points = data_to_do[index]
                points = points.reshape(4,-1,3).astype(np.float32)

                if arp == True:
                  points_arp = np.zeros((4,30000,90))
                  for i_p in range(4):
                    choice = np.random.choice(ct_source_cord.shape[0],100000,replace=False)
                    points_arp[i_p] = ARPD(ct_source_cord[choice],points[i_p])

                  points = torch.from_numpy(points_arp).float().cuda()
                else:
                  points = torch.from_numpy(points).float().cuda()
                points = points.transpose(2,1)

                seg_pred_s = classifier(points)

                seg_pred_choice = seg_pred_s.contiguous().view(-1, cls_num)
                pred_choice = seg_pred_choice.data.max(1)[1]

                if index == len(data_to_do) - 1:
                    pred[-sample_num:] = pred_choice.cpu()
                else:
                    pred[index*sample_num:(1+index)*sample_num] = pred_choice.cpu()
            
            np.savez_compressed(dir_save+name[:-4], ct=ct_cords, seg =pred)


if __name__ == '__main__':
    args = parse_args()
    main(args)
