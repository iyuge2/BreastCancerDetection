import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

class CancerData(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        path, age, her2, p53, label = self.df.iloc[item]
        pic = Image.open(path).convert('RGB')
        if self.transforms:
            pic = self.transforms(pic)
        # handle features
        age = torch.tensor([age/100], dtype=torch.float32) # scale
        her2 = torch.zeros(4).scatter_(0, torch.tensor(her2), 1) # onehot
        p53 = torch.tensor([int(p53)], dtype=torch.float32)
        feature = torch.cat((age, her2, p53), dim=0)
        # id
        id = path.split('/')[-2]
        sample = {
            'id': id,
            'image': pic,
            'feature': feature,
            'label': label-1,
        }
        return sample

def get_images_path(df, mode):
    """
    mode: train / test
    """
    dst_df = pd.DataFrame(columns=['id', 'age', 'her2', 'p53', 'label'])
    num = 0
    for i in range(len(df)):
        cur_id, age, her2, p53, label = df.iloc[i]
        img_dir = os.path.join('data/'+mode+'/images', cur_id)
        cur_paths = glob.glob(img_dir + '/*')
        for cur_path in cur_paths:
            if cur_path.split('/')[-1][0] == '.':
                continue
            dst_df.loc[num] = [cur_path, age, her2, p53, label]
            num += 1
    return dst_df
        

def CancerDataLoader(csv_path, image_size, shuffle=True, batch_size=8, test_size=0.2, num_workers=8):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=99, 
                                        shuffle=shuffle, stratify=df['molecular_subtype'].values)
    train_df = df
    train_df = get_images_path(train_df, 'train_2')
    val_df = get_images_path(val_df, 'train_2')
    # oversample on train dataset
    train_y = train_df['label'].tolist()
    ros = RandomOverSampler(random_state=99)
    tmp_df, _ = ros.fit_sample(train_df, train_y)
    train_df = pd.DataFrame(tmp_df, columns=train_df.columns)
    # transform for images
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    datasets = {
        'train': CancerData(train_df, transforms=transform),
        'val': CancerData(val_df, transforms=transform)
    }
    dataLoaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       shuffle=shuffle,
                       num_workers=num_workers) for ds in datasets.keys()
    }
    return dataLoaders

def CancerDataLoaderForTest(csv_path, image_size, batch_size=8, num_workers=0):
    df = pd.read_csv(csv_path)
    df['label'] = -1 # for use CancerData class
    df = get_images_path(df, 'test_2')
    transform = T.Compose([
        T.Resize(image_size),
        # T.CenterCrop([100, 150]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = CancerData(df, transforms=transform) 
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataLoader

if __name__ == "__main__":
    IMAGE_SIZE = (150, 100)
    csv_path = '/home/iyuge2/Project/BreastCancerDetection/data/train/feats.csv'
    dataLoader = CancerDataLoader(csv_path, IMAGE_SIZE, shuffle=True, batch_size=8, test_size=0.2, num_workers=0)
    for data in dataLoader['train']:
        feature = data['feature']
        image = data['image']
        label = data['label']
        print('test')
