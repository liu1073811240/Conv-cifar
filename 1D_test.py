import torch
import torch.nn as nn
import numpy as np

a = torch.randn(100, 1, 20)  # N, C, L
conv_1d = nn.Conv1d(1, 16, 3, 2, 1)

b = conv_1d(a)
# print(b.shape)

rng = np.random.RandomState(0)
X = 5*rng.randn(100, 1, 10)

print(X)

