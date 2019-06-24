# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class SCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SCNN, self).__init__()
        self.conv1 = self._make_conv(3, 64, 5, 2, 5, 5)
        self.conv2 = self._make_conv(64, 128, 5, 2, 5, 5)
        self.conv3 = self._make_conv(128, 128, 3, 1, (4,6), (4,6))
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def _make_conv(self, in_channel, out_channel, kernel_size, padding, pooling, stride):
        conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pooling, stride=stride)
        )
        return conv
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
    
    def forward(self, x):
        residual = x if self.right is None else self.right(x)
        out = self.left(x) + residual
        return F.relu(out)

class Resnet34(nn.Module):
    def __init__(self, num_classes=110):
        super(Resnet34, self).__init__()
        # self.input_size = (224 , 224, 3)
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        # classifier
        self.fc = nn.Linear(512, 128)
        self.fc2 = nn.Linear(134, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )    
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
    
    def forward(self, x, x2):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = torch.cat((x, x2), dim=1)
        return self.fc2(x)

if __name__ == '__main__':
    model = Net()
    print(model)