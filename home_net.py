import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.conv3 = nn.Linear(64*2, 1)

    def forward(self, x):  # [100, 1, 9]
        y = self.conv1(x)
        # print(y.shape)  # [100, 16, 4]
        y = self.conv2(y)
        # print(y.shape)  # [100, 64, 2]

        y = y.reshape(-1, 64*2)
        y = self.conv3(y)
        # print(y.shape)  # [100, 10]

        return y


# if __name__ == '__main__':
#     a = torch.randn(100, 1, 9)
#     b = Net()
#     b(a)
#


