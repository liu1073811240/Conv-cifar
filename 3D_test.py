import torch
import torch.nn as nn

a = torch.randn(100, 3, 9, 224, 224)  # N, C, D, H, W
conv_3d = nn.Conv3d(3, 16, (3, 3, 3), 2, 1)  # 卷积核大小为3*3，2*2

b = conv_3d(a)
print(b.shape)