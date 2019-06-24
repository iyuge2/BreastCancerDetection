import os
import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch import optim

from model import Resnet34, SCNN
from dataLoader import CancerDataLoader

# change current working dir
os.chdir('/home/iyuge2/Project/BreastCancerDetection')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ModelDict = {
    'RESNET34': Resnet34,
    'SCNN': SCNN,
}


def do_train(model, dataLoader, optimizer, criterion, device, model_name,
             model_save_path='tmp', epoches=100, early_stop=8):
    model.to(device)
    best_f1 = 0.0
    best_epoch = 0
    for epoch in range(0, epoches):
        # train
        train_losses = []
        y_pred, y_true = [], []
        model.train()
        for data in tqdm(dataLoader['train']):
            image = data['image'].to(device)
            feature = data['feature'].to(device)
            label = data['label'].to(device)
            # clear gradient
            optimizer.zero_grad()
            # forward
            outputs = model(image, feature)
            loss = criterion(outputs, label)
            # backward
            loss.backward()
            # update
            optimizer.step()
            train_losses += [loss.item()]
            y_pred += torch.max(outputs.cpu(), 1)[1].tolist()
            y_true += label.cpu().tolist()
        train_acc = accuracy_score(y_pred, y_true)
        train_f1 = f1_score(y_pred, y_true, average='macro')
        train_loss = sum(train_losses) / len(train_losses)
        # val
        val_losses = []
        y_pred, y_true = [], []
        model.eval()
        for data in tqdm(dataLoader['val']):
            image = data['image'].to(device)
            feature = data['feature'].to(device)
            label = data['label'].to(device)
            # predict
            outputs = model(image, feature)
            loss = criterion(outputs, label)
            val_losses += [loss.item()]
            y_pred += torch.max(outputs.cpu(), 1)[1].tolist()
            y_true += label.cpu().tolist()
        val_acc = accuracy_score(y_pred, y_true)
        val_f1 = f1_score(y_pred, y_true, average='macro')
        val_loss = sum(val_losses) / len(val_losses)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            files2remove = glob.glob(os.path.join(model_save_path, 'ep_*'))
            for _i in files2remove:
                os.remove(_i)
            # save model
            torch.save(
                model.cpu().state_dict(),
                os.path.join(model_save_path,
                             f'ep_{epoch}_{model_name}_val_f1_{val_f1:.4f}.pth'))
            model.to(device)
        # print
        print('Epoch %d >> train loss: %.5f, train acc: %.2f, train f1: %.2f, val loss: %.5f, val acc: %.2f, val f1:%.2f'
              % (epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1))
        # early stop
        if epoch - best_epoch > early_stop:
            break


def run(csv_path, image_size, model_name, epochs=200, num_classes=4, early_stop=8):
    """
    model_name: 'DNN', 'RESNET34'
    """
    # dataLoader
    dataLoader = CancerDataLoader(
        csv_path, image_size, shuffle=True, batch_size=16, test_size=0.2, num_workers=8)
    # model
    Model = ModelDict[model_name]
    model = Model(num_classes=num_classes)
    # loss
    criterion = nn.CrossEntropyLoss()
    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)
    # using GPU
    device = torch.device('cuda: 0' if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    do_train(model, dataLoader, optimizer, criterion, device, model_name,
             model_save_path='tmp', epoches=epochs, early_stop=early_stop)


if __name__ == '__main__':
    # train image
    IMAGE_SIZE = (150, 100)
    csv_path = 'data/train_2/feats.csv'
    run(csv_path, IMAGE_SIZE, model_name='RESNET34',
        epochs=200, num_classes=4, early_stop=8)
