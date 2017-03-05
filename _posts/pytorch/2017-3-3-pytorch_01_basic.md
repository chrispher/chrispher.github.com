---
layout: post
title: pytorch初步
category: pytorch
tags: [pytorch]
description: pytorch 基本认识和入门
---

pytorch一出来,就立刻试用了一下. 在过年的期间,用pytorch写了一些代码,感觉非常棒. 也许在工程方面,不是最好的,但是用于research确实很爽.

<!-- more -->

### 目录
{:.no_toc}

* 目录
{:toc}

### 1. pytorch VS tensorflow
不得不提的是tensorflow, 网上有很多的对比, 不再赘述, pytorch还有很长的路要走. tensorflow毕竟有谷歌在后面撑着,而且已经开源了一年, 网上有很多基于tensorflow的再开发和系统的优化, 除了使用tensorflow训练模型之外, 还有tensorflow的server用于提供预估服务, 在企业应用方面, 谷歌是大范围使用了tensorflow, 很多研究院也会使用tensorflow进行研究. 在tensorflow 刚开源的时候, 我就使用了一下tensorflow, 还给公司的同事们做过PPT介绍tensorflow, 但是奇怪的语法着实让人感觉不那么美......当然啦, 自动求导这个奇葩的工具, 给大家灌水带来了非常大的便利. 现在很多深度学习库, 都会带这个功能, 比如mxnet, 以及今天要介绍的pytorch. 当然啦,目前的趋势就是底层都是c/c++, 而上层包装一个Python. 在公司里, 写c++写多了, 感觉用c++做机器学习也没啥不好的, 刚开始学机器学习的, 可都是认为python最好的. 后来才发现, 语言都不是什么事情, 什么合适就用啥!

那么我选择pytorch, 而不是tensorflow的原因是什么? 可能主要是语法吧, 不太喜欢tensorflow那种要定义session, session run以及feed数据的方式吧. 静态图和自动求导, 给调试和测试带来了不少麻烦, 当然我相信以后tensorflow会有一个非常好用的debug工具的. 而pytorch一开始就是以research为主的, 因为平时在家学习一些新的东西, 希望自己能够快速实验一些想法, 所以选择了pytorch为主. 2016年自己,用python或者c++写了不少机器学习算法库, 2017年的话, 就基于pytorch玩一些有意思的东西.

以上是瞎扯部分. 主要想表达: 我们玩玩pytorch吧!

入门的操作可以参考官方的一些[入门教程](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb) 我自己写一点简单的操作, 这些操作在建模和数据预处理的都非常重要.

### 2. 基础操作
基本的加减乘除

``` python
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
dtype = torch.FloatTensor

d = 2
h = 2
# 初始化变量, 可以直接使用 torch.ones(d, h) 或者 torch.randn 进行随机初始化
# 主要目的是, 初始化一个容器, 在容器里放一些数据, 这个好处是可以直接用容器里的数据进行运算和测试
x = Variable(torch.ones(d, h).type(dtype), requires_grad=False)
y = Variable(torch.ones(d, h).type(dtype), requires_grad=False)

# 也可以从numpy中进行初始话
a = np.array([[1, 2], [1, 2]])
b = np.array([[2, 3], [3, 2]])
x = torch.from_numpy(a)
y = torch.from_numpy(b)

# 四则运算, 注意x和y 已经从a 和 b里进行初始化了
print(x, y)            # 分别看看x, y是啥
print(x + y)           # 加法
print(x.add(y))        # 等价于 x + y
print(x * 2 + y)       # 还是加法
print(x * y)           # 对应元素的乘积, 得到 [[2, 6], [3, 4]]
print(x.mul(y))        # 等价于 x * y
print(x.dot(y))        # 点乘, 得到 15 = 2 + 6 + 3 + 4
print(x @ y)           # 矩阵叉乘, 得到[[8, 7], [8, 7]]
print(x.mm(y))         # 等价于 x @ y
print(x / y)           # 除法! 一定小心! 如果打印出来一定能够看到x,y是整形的, 所以结果是0!! 而不是小数

# 是否有下划线的区别
print(x.add(y))       # 输出 x + y, x保持不变
print(x.add_(y))      # 输出 x + y, 同时修改x的值 = x + y, 同理dot, mm, mul等操作!!

```


### 3.函数运算和梯度计算

``` python
# 简单的case
x = Variable(torch.ones(1), requires_grad = True)
y = x ** 2 + 3
y.backward(retain_variables=True)
print(x.grad)  # 梯度值是 2*x = 2
# 复杂一点,
target = torch.FloatTensor([10])
y.backward(target, retain_variables=True)
print(x.grad)  # 梯度值是 2*x = 20, 因为retain的设置为true, grad会加上原来的梯度值2, 结果是22

x = Variable(torch.ones(1), requires_grad = True)
y = x ** 2 + 3
target = torch.FloatTensor([10])
y.backward(target)
print(x.grad)      # 直接设置backward, 得到梯度值 20
y.backward(target) # 会报错, 因为x的grad已经被填充了, 不能再放东西进去了
```

这里只是一个随便的x做的测试, 主要是提醒大家retain_variables的问题, 可能会引起的潜在问题.

从上面可以看出来, pytorch可以任意的定义函数,并且进行梯度的计算, 最重要的是我们可以随便输入一个简单的值, 比如1, 0等等进行forward的验证. 同时,我们也可以设置顶一个值, 进行backward的测试, 来验证我们的模型是否符合预期, 以及建模中经常会遇到的维度不一致导致的叉乘点乘混淆在一起.

### 4.简单的回归方法

官方给出了一个回归的example, 我稍微简化一下, 介绍如下

```
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

feature_num = 3
batch_size = b
w_target = torch.randn(feature_num, 1) * 4   # 初始化一下参数w, 作为我们的目标参数
b_target = torch.randn(1) * 3                # 初始化一下bias项, 作为我们的目标参数

def f(x):
    return x.mm(w_target) + b_target[0]

def print_param(W, b):
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x_{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result

def get_batch(batch_size=32):
    """生成一些训练数据, 用于训练"""
    x = torch.randn(batch_size, feature_num)
    y = f(x)
    return Variable(x), Variable(y)

# 定义模型, 通常情况我们都会用nn库中的一些函数来建模, 利用也有的库, 而不是自己写一个
fc = torch.nn.Linear(w_target.size(0), 1)

# 看一下我们要学习的函数
print('Actual  function:\t%s' %(print_param(w_target.view(-1), b_target)))

for batch_idx in range(100):
    batch_x, batch_y = get_batch()
    # 重置梯度
    fc.zero_grad()
    # Forward 操作
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.data[0]
    # Backward 操作
    output.backward()
    # 随机梯度下降学习
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad.data)
    # 设置停止训练的标准
    if loss < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('Learned function:\t%s' %(print_param(fc.weight.data.view(-1), fc.bias.data)))
print('Actual  function:\t%s' %(print_param(w_target.view(-1), b_target)))

```

这个case中, 我们使用了torch自带的nn中的Linear函数和smooth_l1_loss, 但是我们的随机梯度下降却仍然是自己实现的, 当然啦, pytorch也提供了SGD以及相关的函数实现. 详细的见下一个case吧!


### 5.定义一个神经网络

现在我们来实验一个简单的神经网络吧, 依旧是使用mnist数据集, 这个已经被玩烂的数据集, 仍然是入门和测试的好数据集.

``` python
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 常用的一些操作, 主要是卷积, 池化和全链接, 可以尝试加dropout和batchNorm
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 正常情况下, 我们都会用类进行封装一个网络
net = Net()
# 定义目标函数
criterion = nn.CrossEntropyLoss()
# 定义优化方法, 这里使用SGD + 动量的方法
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练整个网络
for epoch in range(2):
    running_loss = 0.0
    # 这里需要自己搞一下数据 官网提供了一些现成的数据供玩耍
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

```

后面的预估什么的, 都是比较简单的, 详细可以看一下官网的case. 这里重点要强调一点(官网上有, 但是这里没有写出来的):

- 查看参数的方式, `params = list(net.parameters())`
- 调试模型:
    - 可以先使用 `x_t = Variable(torch.randn(1, 1, 32, 32))`, 之后 `out = net(input)` 来看看中间结果是否能够正常运行
    - 同样,我们可以随机一个out, 通过 `net.zero_grad()` 和 `out.backward(torch.randn(1, 10))`来看看误差反馈正常与否
- 查看函数链 `print(loss.creator.previous_functions[0][0]) # Linear` 可以不断看到这个函数的前面的函数是啥

学习一个库, 如果是为了应用的话, 一定要多注意check, 因为你可能不知道有很多用法,就会踩坑!! 当然, 看一些教程啥的, 还是能够帮助你入门的.

但是, 最重要是, 读官方文档, 读官方文档, 读官方文档!!!

通过官方文档, 了解基本的运算和基本的坑, 后面就发挥的你想象开始应用吧. 感兴趣的童鞋, 可以看看torch的实现源码!
