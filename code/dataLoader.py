import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

class CancerData(Dataset):
    def __init__(self,
                data_dir,
                mode = 'train',
                image_size = (100,200),
                images_num = 5,
                transforms = None):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.images_num = images_num
        self.data = pd.read_csv(os.path.join(data_dir, mode, 'feats.csv'))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.mode == 'train':
            id, age, her2, p53, subtype = self.data.iloc[item]
            label = torch.LongTensor([int(subtype)-1])
            # label = torch.zeros(4, dtype=torch.long).scatter_(0, label, 1)
        elif self.mode == 'test':
            id, age, her2, p53 = self.data.iloc[item]
        features_data = torch.Tensor([int(age), int(her2), int(p53)])
        # get images
        images_data = torch.zeros(self.images_num, self.image_size[1], self.image_size[0])
        image_dir = os.path.join(self.data_dir, self.mode, 'images', id)
        images_path = os.listdir(image_dir)
        for i, path in enumerate(images_path):
            if i >= self.images_num:
                break
            if path.split('.')[0] == '': # remove path == '.*'
                continue
            pic = Image.open(os.path.join(image_dir, path)).convert('L')
            pic = pic.resize(self.image_size)
            # if pic.mode == 'RGB':
            #     pic = pic.convert('L')
            if self.transforms:
                pic = self.transforms(pic)
            # images_data[i] = torch.from_numpy(np.array(pic))
            images_data[i] = pic

        # images_data = torch.unsqueeze(images_data, 1)
        if self.mode == 'train':
            return images_data, features_data, label
        else:
            return images_data, features_data

def CancerDataLoader(root_dir, image_size, batch_size=32, num_workers=0, shuffle=True):
    transform = T.Compose([
        # T.Resize(image_size),
        # T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[.5], std=[.5])
    ])
    datasets = {
        'train_data': CancerData(root_dir, mode='train', image_size=image_size, transforms=transform),
        'test_data': CancerData(root_dir, mode='test', image_size=image_size, transforms=transform)
    }
    dataLoaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       shuffle=shuffle,
                       num_workers=num_workers) for ds in datasets.keys()
    }
    return dataLoaders

if __name__ == "__main__":
    IMAGE_SIZE = (200, 100)
    ROOT_DIR = '/home/iyuge2/Project/BreastCancerDetection/data'

    dataLoader = CancerDataLoader(ROOT_DIR, IMAGE_SIZE)
    for data in dataLoader['train_data']:
       img, fea, label = data
       print(img.shape, fea.shape, label.shape)
    
