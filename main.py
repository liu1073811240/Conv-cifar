from Net_conv import Net
import torch.nn as nn
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


if __name__ == '__main__':
    batch_size = 100
    save_params = "./net_params.pth"
    save_net = "./net.pth"

    train_data = datasets.MNIST("./mnist", True, transforms.ToTensor(), download=True)
    test_data = datasets.MNIST("./mnist", False, transforms.ToTensor(), download=True)

    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    # net.load_state_dict(torch.load(save_params))
    # net = torch.load(save_net).to(device)

    # loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()  # 自动对网络输出做softmax， 自动对标签做one-hot
    loss_fn = nn.BCELoss()  # 输出必须经过sigmoid激活，其它规则和MSE一样, 一般用来做置信度损失。
    # loss_fn = nn.BCEWithLogitsLoss()  # 相当于sigmoid+BCELOSS

    optim = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,)

    plt.ion()
    a = []
    b = []
    net.train()

    for epoch in range(2):
        for i, (x, y) in enumerate(train_loader):
            y = torch.zeros(y.size(0), 10).scatter_(1, y.reshape(-1, 1), 1)
            x = x.to(device)
            y = y.to(device)
            out = net(x).reshape(y.size(0), -1)  # 将输出转为N,V结构

            loss = loss_fn(torch.sigmoid(out), y)
            # loss = loss_fn(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 100 == 0:
                plt.clf()
                a.append(i + epoch*(len(train_data) / batch_size))
                b.append(loss.item())
                plt.plot(a, b)
                plt.pause(0.01)

                print("epoch:{}, loss:{:.3f}".format(epoch, loss.item()))

            torch.save(net.state_dict(), save_params)
            # torch.save(net, save_net)

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        y = torch.zeros(y.size(0), 10).scatter_(1, y.reshape(-1, 1), 1)
        x = x.to(device)
        y = y.to(device)

        out = net(x).reshape(y.size(0), -1)
        loss = loss_fn(out, y)

        eval_loss += loss.item()*batch_size

        max_y = torch.argmax(y, 1)
        max_out = torch.argmax(out, 1)
        eval_acc += (max_out == max_y).sum().item()

    mean_loss = eval_loss / len(test_data)
    mean_acc = eval_acc / len(test_data)
    print("mean loss:{}, mean acc:{}".format(mean_loss, mean_acc))








