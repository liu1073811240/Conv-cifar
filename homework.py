'''
作业：利用一维卷积做信号测试
步骤： 生成100-300范围之间的100个随机数据（可以生成更多），并将其可视化出来。


注意：
    1.其中一维卷积核是随机生成出来的。
    2. 100个随机数据就是原始信号，波形图已经固定，不需要打乱去取数据。所以我们应该按
    顺序去取数据。比如说第一次我们拿前面的九个数据作为输入x（：9），第十个数据作为标签y（9：10），
    第二次步长加1，再继续去卷，数据输入为x(1:10), y(10:11), 第三次步长还是为1，数据输入为x(2:11),
    y（11:12），类似这种规律，就能把所有的数据取到，
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from home_net import Net
import random

net = Net()

loss_func = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# rng = np.random.RandomState(0)
# X = 100*rng.randn(100, 1, 9)  # 100个数据，一个通道，长度为9

train_data = np.random.RandomState(0).uniform(200, 300, size=100)
train_data_ = torch.tensor(train_data, dtype=torch.float32)


a = []
b = []
c = []
plt.ion()
for epoch in range(5):
    for i in range(len(train_data_) - 9):  # 循环91次
        x = train_data_[i: i+9]  # 拿前面九个数据
        y = train_data_[i+9: i+10]  # 拿第十个数据作为标签

        xs = torch.reshape(x, [-1, 1, 9])  # N,C, L

        ys = torch.reshape(y, [-1, 1])  # [N, V]
        y1 = ys.cpu().item()
        c.append(y1)
        print(c)
        # exit()

        output = net(xs)
        # print(output)
        # print(output.shape)  # [1, 1]
        out = output.to("cpu").detach()[0][0]
        print(out.item())

        loss = loss_func(output, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        error = loss.item()

        num = (len(train_data) - 9)*epoch + i
        a.append(num)
        b.append(out.item())
        # print(c)
        # print(b)
        # exit()

        plt.clf()
        plt.plot(a, c, c="r", label="label_lines")  # 标签线
        plt.plot(a, b, c="b", label="output_lines")  # 预测线
        plt.legend()
        plt.pause(0.001)
        # print(i)

    torch.save(net.state_dict(), "./params.pth")

plt.ioff()












