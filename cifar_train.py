from torch.utils.data import DataLoader
from full_conv import Net
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

if __name__ == '__main__':
    batch_size = 100
    save_params = "./net_params.pth"
    save_net = "./net.pth"

    train_data = datasets.CIFAR100("./cifar100", True, transforms.ToTensor(), download=True)
    test_data = datasets.CIFAR100("./cifar100", False, transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    # print(train_data.data.shape)  # (50000, 32, 32, 3)
    # print(np.shape(train_data.targets))  # (50000,)
    # print(train_data.classes)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # print(train_data.train)  # True
    # print()
    # print(test_data.data.shape)  # (10000, 32, 32, 3)
    # print(np.shape(test_data.targets))  # (10000,)
    # print(test_data.classes)
    # print(test_data.train)  # False

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    net.load_state_dict(torch.load(save_params))
    # net = torch.load(save_net).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters())

    plt.ion()
    net.train()
    a = []
    b = []
    for epoch in range(10):
        for i, (x, y) in enumerate(train_loader):

            # print(x[0])
            # xs = x[0].data.numpy()  # (3, 32, 32)
            # xs = xs.transpose(1, 2, 0)  # (32, 32, 3)
            #
            # xs = (xs*0.5+0.5)*255
            # img = Image.fromarray(np.uint8(xs))
            # plt.imshow(img)
            # plt.pause(1)

            # y = torch.zeros(y.size(0), 10).scatter_(1, y.reshape(-1, 1), 1)
            x = x.to(device)
            y = y.to(device)

            output = net(x).reshape(y.size(0), -1)
            loss = loss_fn(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 10 == 0:
                plt.clf()
                a.append(i + epoch*(len(train_data) / batch_size))
                b.append(loss.item())
                plt.plot(a, b)
                plt.pause(1)

                print("epoch:{}, loss:{}".format(epoch, loss.item()))

        torch.save(net.state_dict(), save_params)  # 每一轮保存网络参数
        # torch.save(net, save_net)

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        # y = torch.zeros(y.size(0), 10).scatter_(1, y.reshape(-1, 1), 1)
        x = x.to(device)
        y = y.to(device)

        out = net(x).reshape(y.size(0), -1)
        loss = loss_fn(out, y)

        eval_loss += loss.item() * batch_size

        # max_y = torch.argmax(y, 1)
        max_out = torch.argmax(out, 1)
        eval_acc += (max_out == y).sum().item()

    mean_loss = eval_loss / len(test_data)
    mean_acc = eval_acc / len(test_data)
    print("mean loss:{}, mean acc:{}".format(mean_loss, mean_acc))




