import torch.nn as nn
import torch
import torch.utils.data as data
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convf = nn.Sequential(  # torch.Size([100, 3, 32, 32])
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)
            # N,64,14,14
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # N,16,7,7
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)  # N,64,3,3
        )

        self.conv3 = nn.Conv2d(64, 10, 3, 1)  # N,10,1,1

    def forward(self, x):

        y = self.convf(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)

        return y





