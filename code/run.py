import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from dataLoader import CancerDataLoader
from model import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

def do_train(trainLoader, model, epoches=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # using GPU
    device = torch.device('cuda: 0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(0, epoches):
        train_loss = 0.0
        true_sum = 0
        with tqdm(enumerate(trainLoader)) as dd:
            for i, data in dd:
                img, fea, label = data
                img = img.to(device)
                label = label.squeeze().to(device)
                # clear gradient
                optimizer.zero_grad()
                # forward
                outputs = model(img)
                loss = criterion(outputs, label)
                # backward
                loss.backward()
                # update
                optimizer.step()
                train_loss += loss.item()
                y_pred = torch.max(outputs, 1)[1]
                true_sum += (y_pred == label).sum()
        # print
        print('Epoch %d >> train loss: %.5f, train acc: %.2f' %(epoch, train_loss, true_sum.item() / 200.0))

def do_test():
    pass

def run(data_dir, image_size, epochs=200):
    dataLoader = CancerDataLoader(data_dir, image_size, batch_size=32, num_workers=8, shuffle=True)
    model = Net(4)
    trainLoader = dataLoader['train_data']
    do_train(trainLoader, model, epoches=epochs)

if __name__ == '__main__':
    IMAGE_SIZE = (200, 100)
    ROOT_DIR = '/home/iyuge2/Project/BreastCancerDetection/data'
    run(ROOT_DIR, IMAGE_SIZE, 200)
