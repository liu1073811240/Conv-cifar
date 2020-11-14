import torch.nn as nn
import torch
import torch.utils.data as data
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convf = nn.Sequential(  # torch.Size([100, 3, 32, 32])
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # N,64,28,28
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # N,16,14,14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # N,64,7,7
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),  # N,128, 3, 3
            # nn.AvgPool2d((7, 7), 1)  # 全局平均池化,形状变为N,128,1,1
        )

        self.conv4 = nn.Conv2d(128, 100, 3, 1)  # N,10,1,1

    def forward(self, x):

        y = self.convf(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)

        return y





