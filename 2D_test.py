import torch
import torch.nn as nn

a = torch.randn(100, 1, 20, 20)  # N, C, H, W
conv_2d = nn.Conv2d(1, 16, (3, 3), 2, 1)  # 卷积核大小为3*3

b = conv_2d(a)
print(b.shape)