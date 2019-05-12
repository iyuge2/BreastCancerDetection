# coding: utf-8
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_data(
            data_dir = '..',
            mode = 'train',
            return_dict = False,
            image_size = (200,100)):
    """
    data_dir: data root dir
    mode: train/test
    return_dict: if True, return whole dict, else return data and label as list
    image_size: (Width, Height)
    """
    data = {}
    features_data, images_data, labels = [], [], []
    df = pd.read_csv(os.path.join(data_dir, mode, 'feats.csv'))
    for i in tqdm(range(len(df))):
        cur_data = {}
        # get features and label
        id, age, her2, p53, subtype = df.iloc[i]
        cur_data['features'] = np.array([age, her2, int(p53)], dtype='int32')
        features_data.append(np.array([age, her2, int(p53)], dtype='int32'))
        cur_data['label'] = int(subtype)
        labels.append(int(subtype))
        # get images
        images = []
        image_dir = os.path.join(data_dir, mode, 'images', id)
        images_path = os.listdir(image_dir)
        for path in images_path:
            if path.split('.')[0] == '': # remove path == '.*'
                continue
            pic = Image.open(os.path.join(image_dir, path))
            pic = pic.resize(image_size)
            images.append(np.array(pic))
        cur_data['images'] = np.array(images)
        images_data.append(np.array(images))
#         if len(images) == 0:
#             print(id)
        # save data
        data[id] = cur_data
    if(return_dict):
        return data
    else:
        return features_data, images_data, labels

root_dir = os.path.dirname(os.path.abspath(__file__))
print(root_dir)
train_features, train_images, train_labels = load_data(data_dir = os.path.join(root_dir, '../data'), mode='train', return_dict=False)
# print(train_features.shape, train_images.shape, train_labels.shape)
print('pass')