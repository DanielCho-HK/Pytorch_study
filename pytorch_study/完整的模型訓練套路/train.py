import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *




#準備數據集
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
# print(train_data_size)
# print(test_data_size)

#加載數據集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#創建網絡模型
net = Daniel()

#損失函數
loss_fn = nn.CrossEntropyLoss()

#優化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


#設置訓練網絡的一些參數
total_train_step = 0
total_test_step = 0

epoch = 10

#添加tensorboard
writer = SummaryWriter('./logs_train')



for i in range(epoch):

    print("第{}輪訓練開始".format(i + 1))

    #訓練步驟
    net.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("訓練次數：{}，誤差爲：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #驗證步驟
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整體驗證集上的誤差： {}".format(total_test_loss))
    print("整體驗證集上的準確率： {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(net.state_dict(), "daniel_{}.pth".format(i+1))
    print("模型已保存")
writer.close()
























