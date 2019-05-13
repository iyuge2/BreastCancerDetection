# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # convlution
        self.conv1 = nn.Conv2d(5, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        # dense
        self.fc1 = nn.Linear(20680, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # conv -> activation -> pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # flatten -> dense
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

if __name__ == '__main__':
    model = Net()
    print(model)