import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from PIL import Image

import torch
from torchvision import transforms as T

from dataLoader import CancerDataLoaderForTest
from model import Resnet34

os.chdir('/home/iyuge2/Project/BreastCancerDetection') # change current working dir
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def do_eval(csv_path, dst_path, pth_path, image_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model
    model = Resnet34(num_classes=4)
    # 加载模型
    model.load_state_dict(torch.load(pth_path))
    model.to(device).eval()
    # load data
    num = 0
    tmp_df = pd.DataFrame(columns=['id', 'label'])
    loader = CancerDataLoaderForTest(csv_path, image_size, num_workers=8)
    for m, data in enumerate(tqdm(loader)):
        image = data['image'].to(device)
        feature = data['feature'].to(device)
        id = data['id']
        outputs = model(image, feature)
        y_pred = torch.max(outputs.cpu(), 1)[1].tolist()
        for i in range(len(y_pred)):
            tmp_df.loc[num] = [id[i], y_pred[i]]
            num += 1
    # get results for push
    dst_df = pd.DataFrame(columns=['id', 'label'])
    df = pd.read_csv(csv_path)
    for i in range(len(df)):
        id = df.iloc[i]['id']
        true_label = tmp_df.loc[tmp_df['id']==id]['label'].value_counts().index[0] + 1
        dst_df.loc[i] = [id, true_label]
    dst_df.to_csv(dst_path, header=None, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_path', default='data/train/feats.csv',
                        help='path of input feats.')
    parser.add_argument('--pred_path', default='result.csv',
                        help='path of results.')
    parser.add_argument('--pth_path', default='',
                        help='path of pretrained model.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    IMAGE_SIZE = (150, 100)
    csv_path = args.feats_path
    dst_path = args.pred_path
    pth_path = args.pth_path
    do_eval(csv_path, dst_path, pth_path, IMAGE_SIZE)
