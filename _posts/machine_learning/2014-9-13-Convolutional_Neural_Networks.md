---
layout: post
title: Convolutional Neural Networks
category: 机器学习
tags: [深度学习]
description: 讨论了卷积神经网络(CNN)的基本概念和实现，并且针对某些CNN结构做具体的分析。CNN的发展也是经历了很长时间，不同的人也会有不同的实现方法和技巧，但是基本概念是相同的，这里以LeNet5为主要参照。
---

CNN ([Convolutional Neural Networks](http://en.wikipedia.org/wiki/Convolutional_neural_network)) 是神经网络的一种，也是受启发于生物学，广泛的应用与图像分类等任务中。 这篇文档主要讨论CNN的基本概念和实现，并且针对某些CNN结构做具体的分析。CNN的发展也是经历了很长时间，不同的人也会有不同的实现方法和技巧，但是基本概念是相同的，这里以LeNet5为主要参照，词语表达以图像处理为基础。本文为个人的知识管理而进行资料整理，并不应用于商业目的，仅供学习交流。

<!-- more -->

###目录
{:.no_toc}

* 目录
{:toc}

###1.概述
卷积神经网络是人工神经网络的一种，已成为当前语音分析和图像识别领域的研究热点。它的权值共享网络结构使之更类似于生物神经网络，降低了网络模型的复杂度，减少了权值的数量。该优点在网络的输入是多维图像时表现的更为明显，使图像可以直接作为网络的输入，避免了传统识别算法中复杂的特征提取和数据重建过程。卷积网络是为识别二维形状而特殊设计的一个多层感知器，这种网络结构对平移、比例缩放、倾斜或者共他形式的变形具有高度不变性[$$^6$$](http://blog.csdn.net/zouxy09/article/details/8781543)。
典型的CNN是由卷积层(Convolution Layers)和子抽样层(Sub-sampling Layers)构成，由这两层组合成多层CNN网络，用于特征提取和表征，我们可以在网络最后增加分类层达到分类器的效果。

###2.卷积层和池化层
这部分详细的说一下卷积层和池化层（子抽样层的一种典型），并根据这两层的特点说明CNN中的**权值共享**和**部分联通**。

####2.1 卷积层
自然图像有其固有特性，也就是说，图像的一部分的统计特性与其他部分是一样的。这也意味着我们在这一部分学习的特征也能用在另一部分上，所以对于这个图像上的所有位置，我们都能使用同样的学习特征[$$^1$$](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)。
更恰当的解释是，当学习到一个大小为 5x5 特征时（也可以称之为filter，以下均用filter表示，可以翻译为滤波器或者卷积核），把该特征作为探测器，应用到这个图像的任意地方中去。即用从 5x5 的filter跟原本的大尺寸图像作卷积，从而对这个大尺寸图像上的任一位置获得一个不同特征的激活值。注意，我们这里直接使用**卷积**操作这个词，与一般所说的特征提取、感受野等其他词汇不同，这里更倾向于图像处理！

下面解释一下**卷积**操作：假设已经学习到了一个3×3的filter(filter)，为了得到卷积特征，需要对原始 5x5 的图像的每个 3x3 的小块图像区域都进行卷积运算。也就是说，抽取 3x3 的小块区域，并且从起始坐标开始依次标记为（1，1），（1，2），...，一直到（3，3），然后对抽取的区域逐个运行卷积得到大小为 3×3 卷积特征（3 = (5-3)/1 + 1）。
<img src="/images/deeplearning/cnn_Convolution_schematic.gif" height="100%" width="100%">

**注意**：图示中的卷积操作，每次只移动一个像素，在某些大图像操作中，比如200×200像素的图像中，filter大小为10×10，我们也可以设定每次移动的步伐（有的称之为stride，有的称之为sub-sample）为10，那么最终得到的卷积特征大小为19×19（19=(200-10)/10 + 1）。

这里需要着重解释一下什么是**参数减少**和**权值共享**。比如我们有1000x1000像素的图像，假设有1百万个隐层神经元，那么他们全连接的话（每个隐层神经元都连接图像的每一个像素点），就有 $$ 1000x1000x1000000 = 10^12$$ 个连接，也就是$$10^12$$个权值参数。然而图像的空间联系是局部的，就像人是通过一个局部的感受野去感受外界图像一样，每一个神经元都不需要对全局图像做感受，每个神经元只感受局部的图像区域，然后在更高层，将这些感受不同局部的神经元综合起来就可以得到全局的信息了。这样，我们就可以减少连接的数目，也就是减少神经网络需要训练的权值参数的个数了。假如局部感受野(filter)是10x10，隐层每个感受野只需要和这10x10的局部图像相连接，所以1百万个隐层神经元就只有一亿个连接，即$$10^8$$个参数。
<img src="/images/deeplearning/cnn_w_share.jpg" height="100%" width="100%">

上面说的是**参数减少**，而CNN是在参数减少之上再减少的**权值共享**。隐含层的每一个神经元都连接10x10个图像区域，也就是说每一个神经元存在10x10=100个连接权值参数。那如果我们每个神经元这100个参数是相同的呢？也就是说每个神经元用的是同一个filter去卷积图像。这样我们就只有100个参数了。

但是这样共享一个权值，只能提取了一种特征？如果我们用多种filter的话，就能提出图像的不同的特征。所以假设我们加到100种filter，每种filter的参数不一样，表示它提出输入图像的不同特征，这样每种filter去卷积图像就得到对图像的不同特征的表达，我们称之为Feature Map。所以100种filter就有100个Feature Map。这100个Feature Map就组成了一层神经元。那这一层有多少个参数了？100种filterx每种filter共享100个参数=100x100=10K，也就是1万个参数。见下图右：不同的颜色表达不同的滤波器。
<img src="/images/deeplearning/cnn_w_share2.jpg" height="100%" width="100%">

####2.2 池化层
在通过卷积获得了特征 (features) 之后，下一步我们希望利用这些特征去做分类。理论上讲，人们可以用所有提取得到的特征去训练分类器，例如 softmax 分类器，但这样做面临计算量的挑战。例如：对于一个 96X96 像素的图像，假设我们已经学习得到了400个定义在8X8输入上的特征，每一个特征和图像卷积都会得到一个 (96 − 8 + 1) × (96 − 8 + 1) = 7921 维的卷积特征，由于有 400 个特征，所以每个样例 (example) 都会得到一个 892 × 400 = 3,168,400 维的卷积特征向量。学习一个拥有超过 3 百万特征输入的分类器十分不便，并且容易出现过拟合 (over-fitting)。

为了解决这个问题，首先回忆一下，我们之所以决定使用卷积后的特征是因为图像具有一种“静态性”的属性，这也就意味着在一个图像区域有用的特征极有可能在另一个区域同样适用。因此，为了描述大的图像，一个很自然的想法就是对不同位置的特征进行聚合统计，例如，人们可以计算图像一个区域上的某个特定特征的平均值 (或最大值)。这些概要统计特征不仅具有低得多的维度 (相比使用所有提取得到的特征)，同时还会改善结果(不容易过拟合)。这种聚合的操作就叫做池化 (pooling)，有时也称为平均池化或者最大池化 (取决于计算池化的方法)。

pooling的本质是一种局部特征的表达。max pooling的意思就是用图像某一区域像素值的最大值来表示该区域的特征，而mean pool的意思用图像某一区域像素值的均值来表示该区域的特征。这两个pooling操作都提高了提取特征的不变性，而特征提取的误差主要来自两个方面：（1）邻域大小受限造成的估计值方差增大；（2）卷积层参数误差造成估计均值的偏移。一般来说，mean-pooling能减小第一种误差，更多的保留图像的背景信息，max-pooling能减小第二种误差，更多的保留纹理信息。在图像处理中，使用max pooling多于mean pooling。


下图显示池化如何应用于一个图像的四块不重合区域。
<img src="/images/deeplearning/cnn_Pooling_schematic.gif" height="100%" width="100%">

如果人们选择图像中的连续范围作为池化区域，并且只是池化相同(重复)的隐藏单元产生的特征，那么，这些池化单元就具有平移不变性 (translation invariant)。这就意味着即使图像经历了一个小的平移之后，依然会产生相同的 (池化的) 特征。在很多任务中 (例如物体检测、声音识别)，我们都更希望得到具有平移不变性的特征，因为即使图像经过了平移，样例(图像)的标记仍然保持不变。例如，如果你处理一个MNIST数据集的数字，把它向左侧或右侧平移，那么不论最终的位置在哪里，你都会期望你的分类器仍然能够精确地将其分类为相同的数字[$$^1$$](http://deeplearning.stanford.edu/wiki/index.php/Pooling)。

###3.稀疏连接与LeNet5说明
根据上面的说明，我们基本上知道了卷积神经网络中各个操作的基本含义：即卷积层做的什么操作，池化层做的什么操作。但是，这些层是如何连接在一起的呢？下图是LeNet5用于手写字母识别的结构图：

<img src="/images/deeplearning/cnn_lenet5.png" height="100%" width="100%">

LeNet-5[$$^4$$](http://enpub.fulton.asu.edu/cseml/summer08/papers/cnn-pieee.pdf)共有7层(不包含输入层)，这里输入图像大小是32×32，经过大小为6个5×5的filter卷积操作后，得到C1层(complex cells)，C1层就有6个feature maps（等于filter数）,每个feature map的大小是28×28(stride=1, 28 = (32-5)/1 + 1 )，这一层有多少参数？每个filter5×5=25个参数，一共6层（每层feature map还有一个偏置项，在后面的推导公式中会具体提到），那么C1层一共 6×5×5+6 = 156个参数，一共156*(28*28)=122,304个连接。之后跟着S2层是一个下采样层(池化层、simple cells)，pooling大小是2×2，这里的feature maps数目不变，14 = 28 / 2 ，feature maps的每个单元与C1中相对应特征图的2×2邻域相连接。S2层每个单元的4个输入相加，乘以一个可训练参数，再加上一个可训练偏置。结果通过sigmoid函数（或其他函数）计算。可训练系数和偏置控制着sigmoid函数的非线性程度。如果系数比较小，那么运算近似于线性运算，亚采样相当于模糊图像。如果系数比较大，根据偏置的大小亚采样可以被看成是有噪声的“或”运算或者有噪声的“与”运算。每个单元的2×2感受野并不重叠，因此S2中每个特征图的大小是C1中特征图大小的1/4（行和列各1/2）。S2层有12个可训练参数（6个权值和6个偏置项）和5880个连接。
<img src="/images/deeplearning/cnn_s2.jpg" height="100%" width="100%">

上图说明：第一阶段是输入的图像，后面的阶段就是卷积特征map，然后加一个偏置$$b_x$$，得到卷积层$$C_x$$。子采样过程包括：每邻域四个像素求和变为一个像素，然后通过标量$$W_{x+1}$$加权，再增加偏置$$b_{x+1}$$，然后通过一个sigmoid激活函数，产生一个大概缩小四倍的特征映射图$$S_{x+1}$$。
所以从一个平面到下一个平面的映射可以看作是作卷积运算，S-层可看作是模糊滤波器，起到二次特征提取的作用。隐层与隐层之间空间分辨率递减，而每层所含的平面数递增，这样可用于检测更多的特征信息。
注意：这里的pooling操作(subsample)是，取前一层的2×2区域的和。通常情况下，我们会选择一个2×2区域里的最大值，也有直接不使用训练系数的，直接sigmoid操作。

C3层也是一个卷积层，它同样通过5x5的卷积核去卷积层S2，然后得到的特征map就只有10x10个神经元，但是它有16种不同的卷积核，所以就存在16个特征map了。这里需要注意的一点是：C3中的每个特征map是连接到S2中的所有6个或者几个特征map的，表示本层的特征map是上一层提取到的特征map的不同组合。这样做的目的是不完全的连接机制将连接的数量保持在合理的范围内。其次，也是最重要的，其破坏了网络的对称性。由于不同的特征图有不同的输入，所以迫使他们抽取不同的特征（希望是互补的）。

举个例子而言，如下图所示，m层是由m-1层相邻的三个featured map组合而成，即由三个feature map 卷积组合（求和）得到，这称之为稀疏连接（Sparse Connectivity）。
<img src="/images/deeplearning/cnn_sparse_1D_nn.png" height="100%" width="100%">

而LeNet中的S2层到C3层的连接方式如下：
<img src="/images/deeplearning/cnn_lenet5_c3.png" height="100%" width="100%">

从上面看，一共有3×6 + 4×9 + 6 = 60个filter，一共有78×5×5+16 = 1516个参数，一共1516×10×10=151600个连接。 S4层是一个下采样层，由16个5×5大小的特征图构成。特征图中的每个单元与C3中相应特征图的2×2邻域相连接，跟C1和S2之间的连接一样。S4层有32个可训练参数（每个特征图1个因子和一个偏置）和2000个连接。 C5层是一个卷积层，有120个特征图。每个单元与S4层的全部16个单元的5×5邻域相连。由于S4层特征图的大小也为5×5（同滤波器一样），故C5特征图的大小为1×1，这构成了S4和C5之间的全连接。之所以仍将C5标示为卷积层而非全相联层，是因为如果LeNet-5的输入变大，而其他的保持不变，那么此时特征图的维数就会比1×1大。C5层有48120个可训练连接。
F6层有84个单元（之所以选这个数字的原因来自于输出层的设计），与C5层全相连。有10164个可训练参数。如同经典神经网络，F6层计算输入向量和权重向量之间的点积，再加上一个偏置。然后将其传递给sigmoid函数产生单元i的一个状态。最后，输出层由欧式径向基函数（Euclidean Radial Basis Function）单元组成，每类一个单元，每个有84个输入。换句话说，每个输出RBF单元计算输入向量和参数向量之间的欧式距离。输入离参数向量越远，RBF输出的越大。一个RBF输出可以被理解为衡量输入模式和与RBF相关联类的一个模型的匹配程度的惩罚项。用概率术语来说，RBF输出可以被理解为F6层配置空间的高斯分布的负log-likelihood。给定一个输入模式，损失函数应能使得F6的配置与RBF参数向量（即模式的期望分类）足够接近。这些单元的参数是人工选取并保持固定的（至少初始时候如此）。这些参数向量的成分被设为-1或1。虽然这些参数可以以-1和1等概率的方式任选，或者构成一个纠错码，但是被设计成一个相应字符类的7*12大小（即84）的格式化图片。这种表示对识别单独的数字不是很有用，但是对识别可打印ASCII集中的字符串很有用[$$^6$$](http://blog.csdn.net/zouxy09/article/details/8781543)。

###4.theano实现
上面都是从概念上去理解CNN，以及CNN的结构解释。在常见的论文中，一般都是对CNN的结构进行变化，比如增加层数、使用金字塔式训练，还有一些trick。接下来是CNN的数学方面的推导，这个在不同的人实现方式不一样，也会有一些细微的差别。比如我们看到LeNet5中的稀疏连接是手动指定的，有的实现是自己学习这种连接方式。而有些CNN中的filter是通过预训练得到，有些是直接手动指定，当然多数还是通过BP算法（这里不再细述BP算法）学习到的。

我们从matlab代码中能够更加细致的看到CNN的一步步实现过程。但是，对于一些卷积、池化等细节操作，我们仍然可以选择使用一些封装好的函数。这里，我们采用theano来学习CNN的code[$$^5$$](http://www.deeplearning.net/tutorial/lenet.html#lenet)。

{% highlight python %}
class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        
        # 初始化参数，根据随机参数rng 随机生成w和b
        # 注意这里随机生成数时，采用的技巧，设定了一些上下限
        fan_in =  numpy.prod(filter_shape[1:])
        W_values = numpy.asarray(rng.uniform(
              low=-numpy.sqrt(3./fan_in),
              high=numpy.sqrt(3./fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W')

        # 每个feature map 都有一个偏置项
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input, self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will thus
        # be broadcasted across mini-batches and feature map width & height
        # 这里对pooled_out并没有乘以一个vector，可以增加
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
{% endhighlight %}

模型学习的过程如下：
{% highlight python %}
# 设定学习速率和随机种子
learning_rate = 0.1
rng = numpy.random.RandomState(23455)

# 输入图像大小为28×28
ishape = (28, 28)  # this is the size of MNIST images
batch_size = 20  # sized of the minibatch

# allocate symbolic variables for the data
x = T.matrix('x')  # rasterized images
y = T.lvector('y')  # the labels are presented as 1D vector of [long int] labels

##############################
# BEGIN BUILDING ACTUAL MODE
##############################

# Reshape matrix of rasterized images of shape (batch_size,28*28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((batch_size,1,28,28))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
# maxpooling reduces this further to (24/2,24/2) = (12,12)
# 4D output tensor is thus of shape (20,20,12,12)
# filter_shape = （该层maps数, 上一层maps数目, 卷积核宽度x, 卷积核长度y）
layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(20, 1, 5, 5), poolsize=(2, 2))

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12 - 5 + 1, 12 - 5 + 1)=(8, 8)
# maxpooling reduces this further to (8/2,8/2) = (4, 4)
# 4D output tensor is thus of shape (20,50,4,4)
# 这里的filter并没有实现稀疏连接，这个是map与map之间的全连接
# 把map想象成一个神经元，想象一下
layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
        image_shape=(batch_size, 20, 12, 12),
        filter_shape=(50, 20, 5, 5), poolsize=(2, 2))

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20, 32 * 4 * 4) = (20, 512)
layer2_input = layer1.output.flatten(2)

# 引入一层全连接层，并且加上一层softmax用于分类
# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input,
                     n_in=50 * 4 * 4, n_out=500,
                     activation=T.tanh    )

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)


# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
test_model = theano.function([x, y], layer3.errors(y))

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
# 计算偏导，这是theano的特色，不用自己实现复杂的导数计算。
# 想知道具体如何计算，可以参考matlab代码
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by SGD
# Since this model has many parameters, it would be tedious to manually
# create an update rule for each model parameter. We thus create the updates
# dictionary by automatically looping over all (params[i],grads[i])  pairs.
updates = []
for param_i, grad_i in zip(params, grads):
    updates.append((param_i, param_i - learning_rate * grad_i))
train_model = theano.function([index], cost, updates = updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})
{% endhighlight %}

完整代码，可以去参考5中下载。

####5.参考资料
- [1] [stanford UFLDL trtorial](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)
- [2] [Convolutional_neural_network](http://en.wikipedia.org/wiki/Convolutional_neural_network)
- [3] [Notes on Convolutional Neural Networks](http://cogprints.org/5869/1/cnn_tutorial.pdf)
- [4] [Gradient-based learning applied to documents recognition](http://enpub.fulton.asu.edu/cseml/summer08/papers/cnn-pieee.pdf)
- [5] [theano Lenet](http://www.deeplearning.net/tutorial/lenet.html#lenet)
- [6] [Deep Learning（深度学习）学习笔记整理系列之（七）](http://blog.csdn.net/zouxy09/article/details/8781543)

