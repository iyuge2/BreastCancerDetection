import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
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

if __name__ == '__main__':
    IMAGE_SIZE = (150, 100)
    csv_path = 'data/test_2/feats.csv'
    dst_path = 'result.csv'
    pth_path = '/home/iyuge2/Project/BreastCancerDetection/tmp/ep_12_RESNET34_val_f1_0.5164.pth'
    do_eval(csv_path, dst_path, pth_path, IMAGE_SIZE)
