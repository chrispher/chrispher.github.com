---
layout: post
title: 深度学习框架caffe
category: 深度学习
tags: [环境搭建, caffe]
description: 关于深度学习框架caffe的使用文档。
---

本文主要介绍深度学习框架caffe的介绍和入门级使用，以及caffe的python的一些使用说明和注意点，主要参考[caffe官方文档](http://caffe.berkeleyvision.org/tutorial/)。

<!-- more -->

### 目录
{:.no_toc}

* 目录
{:toc}

### 1、caffe简介
Caffe，全称Convolutional Architecture for Fast Feature Embedding，是一个计算CNN相关算法的框架。caffe是一个清晰，可读性高，快速的深度学习框架。作者是贾扬清,加州大学伯克利的ph.D。


### 2、模型训练
这里主要介绍caffe训练CNN的模型。官网给了一些入门的例子，以人脸识别数据为例，

#### 2.1、数据准备

首先准备原始的训练数据和验证数据集，采用分类的方式训练CNN。我们的原始数据是按照类别放在一起，即facenet文件下是很多人，每个人一个文件夹，用于存放所以照片。之后处理成如下文件结构如下：

``` python
- train/ #存放训练数据
    -n01440765/ #每个人一个文件夹
        n01440765_1.jpg
        ...

- val/  #测试数据，放一起即可
    n01440764_14.jpg
    ...
det_synset_words.txt # 存放了原始文件夹名与编码后文件夹名对应关系
synset.txt           # 存放了所有的编码后文件夹名字
synset_words.txt     # 存放了所有的编码后文件夹名及其对应的原文件夹名
train.txt            # 存放训练数据路径以及对应的类别
test.txt             # 存放测试数据路径以及对应的类别
val.txt              # 存放验证数据路径以及对应的类别

```

这里编码的好处是防止caffe无法识别原始文件名，而且需要把类别处理成整数型的数据。把原始图片处理成caffe待使用的数据。

> 更新于2015-11-6 增加数据预处理的脚本，数据预处理的主要思路就是把数据整理成你模型需要输入的格式和样式。

``` python
# -*- coding:utf-8 -*-

import os
import shutil
import numpy as np

base_dir = 'facedata/'    # 原始数据路径
target_dir = 'facenet/'   # 目标数据路径

# 把原始的文件名，更改为新的编码，这里可以随机设定
# 我们从n01440764开始计数，共计60000人（至少要大于类别数）
synset = ['n0'+str(1440764 + i) for i in xrange(6000)]

if os.path.exists(target_dir):
    pass
else:
    os.makedirs(target_dir)

metanames = []
label = -1
for sub_dir in os.listdir(base_dir):
    # 如果样本数不足10，那么不记录该类
    if len(os.listdir(base_dir + sub_dir)) < 10:
        continue
    else:
        pass
    label += 1
    for name in os.listdir(base_dir + sub_dir):
        metanames.append(str(label)+','+ sub_dir+','+name)

# 输出总的人数(类别数)
print 'number of metanames:', len(metanames), label+1

# 把数据乱序，选80%用于训练，20%用于测试（注意，为了尽可能的多训练数据，我们的val数据和test数据是相同的，其实是不需要一致）
np.random.seed(234)
np.random.shuffle(metanames)
# for train
totle_num = len(metanames)
train_num = int(totle_num*0.8)
test_num = totle_num - train_num

print 'number of metanames:', len(metanames)
print 'number of train:', train_num

f0 = open(target_dir+'train.txt', 'w')

labels = []
det_synset_words = set()
print 'prepare the trainning data...'
for meta in metanames[0:train_num]:
    label,sub_dir0,name = meta.strip().split(',')
    filename = base_dir + sub_dir0 + '/' + name
    sub_dir = synset[int(label)]
    det_synset_words.add((sub_dir, sub_dir0))
    name = sub_dir + '_' + name.split('_')[1]
    targetname = target_dir + 'train/' + sub_dir + '/' + name
    if os.path.exists(target_dir + 'train/' + sub_dir):
        pass
    else:
        os.makedirs(target_dir + 'train/' + sub_dir)
    f0.write(sub_dir + '/' + name + ' ' + label)
    f0.write('\n')
    labels.append(label)
    shutil.copy(filename, targetname)
f0.close()

# 记录 det_synset_words
f3 = open(target_dir+'det_synset_words.txt', 'w')
f3_2 = open(target_dir+'synset_words.txt', 'w')
det_synset_words = sorted(list(det_synset_words), key=lambda x:x[0])
for i in det_synset_words:
    f3.write(' '.join(i) + '\n')
    f3_2.write(' '.join(i) + '\n')
f3.close()
f3_2.close()

synset_word = []
for i in set(labels):
    synset_word.append(synset[int(i)])

synset_word = sorted(synset_word)
f4 = open(target_dir+'synset.txt', 'w')
f4.write('\n'.join(synset_word))
f4.close()

# 输出实际训练的类别数
print 'num of classes for train: ', len(set(labels))

# 生成测试数据
f1 = open(target_dir+'val.txt','w')
f2 = open(target_dir+'test.txt','w')

if os.path.exists(target_dir + 'val/'):
    pass
else:
    os.makedirs(target_dir + 'val/')

labels = []
print 'prepare the testing data...'
for meta in metanames[train_num:train_num+test_num]:
    label,sub_dir0,name = meta.strip().split(',')
    filename = base_dir + sub_dir0 + '/' + name
    sub_dir = synset[int(label)]
    name = sub_dir + '_' + name.split('_')[1]
    # targetname = target_dir + 'val/' + sub_dir + '/' + name
    targetname = target_dir + 'val/' + name

    f1.write(name + ' ' + label)
    f1.write('\n')
    f2.write(name + ' ' + '0')
    f2.write('\n')
    labels.append(label)
    shutil.copy(filename, targetname)
f1.close()
f2.close()
print 'num of classes for test: ', len(set(labels))

```

> 这个脚本主要是配合imagenet做的脚本，实际中有很多文件是不需要写入的。而Imagenet这么做的主要理由是在可视化的时候，可以直接看到name等信息。实际，在某些分类中，可能并不关心label的实际name。

在数据预处理完成之后，我们使用caffe提高的imagenet的数据生成脚`creat_imgnet.sh`本来生成训练数据，注意在caffe路径下使用sh命令，否则里面的一些引用路径会有报错，需要再配置路径。对于`creat_imgnet.sh`文件，需要注意路径的配置，这里列出一部分，其他的可以参照这些进行修改。主要如下:

``` python
EXAMPLE=facenet/face_256_256_31w_alax          # 生成模型训练数据文化夹
TOOLS=build/tools                              # caffe的工具库，不用变

DATA=/home/face/facenet/                   # python脚步处理后数据路径
TRAIN_DATA_ROOT=/home/face/facenet/train/  #待处理的训练数据
VAL_DATA_ROOT=/home/face/facenet/val/      # 带处理的验证数据

RESIZE_HEIGHT=256                              # 把数据resize到模型输入需要的大小

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \                          # 训练数据文件名
    $EXAMPLE/face_train_lmdb                   # 生成训练数据，使用lmdb存储

```

之后使用imagenet下的`make_imagenet_mean.sh`生成均值数据，同样需要**注意修改路径**，以及在caffe路径下使用sh命令。


这里我们也可以自己写脚本生成caffe需要的格式类型。

> 更新与2015-11-6：增加levelDB的格式，这里的输入是三张照片，对应的是一个label的。类似于Triplet Loss的输入。注意：这里的三张照片是随机选取的，并没有完整的遍历所有的，而且生成的数量比较少。这个只是例子，需要自己改动来实现自己需要的脚本，比如多label的或者等等。

``` python

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import caffe
import leveldb
import os


def deal_img(fpath):
    image = caffe.io.load_image(fpath)
    # Reshape image
    image = image[:, :, (2, 1, 0)]
    image = image.transpose((2, 0, 1))
    image = image.astype(np.uint8, copy=False)
    return image


def write_levelDB(dbname, images_lists):
    db = leveldb.LevelDB(dbname, create_if_missing=True,
                         error_if_exists=True, write_buffer_size=268435456)
    wb = leveldb.WriteBatch()

    for count, pic in enumerate(images_lists):
        f1, label1 = pic[0], pic[1]  # data_pos
        f2, label2 = pic[2], pic[3]  # data_anc
        f3, label3 = pic[4], pic[5]  # data_neg

        img1 = deal_img(f1)
        img2 = deal_img(f2)
        img3 = deal_img(f3)

        image = np.vstack((img1, img2, img3))
        label = label1

        # Load image into datum object
        db.Put('%08d_%s' % (count, file), datum.SerializeToString())

        if count % 1000 == 0:
            # Write batch of images to database
            db.Write(wb)
            del wb
            wb = leveldb.WriteBatch()
            print('Processed %i images.' % count)

    if count % 1000 != 0:
        # Write last batch of images
        db.Write(wb)
        print('Processed a total of %i images.' % count)
    else:
        print('Processed a total of %i images.' % count)


fpath = "/home/facenet/"
images_dic = {}
for n, name in enumerate(os.listdir(fpath)):
    np.random.seed(1337)
    pics = [os.path.join(fpath, name, i)
            for i in os.listdir(os.path.join(fpath, name))]
    np.random.shuffle(pics)
    images_dic[n] = pics

images_list = []
num = len(images_dic.keys())
print(n, num)
for name, pics in images_dic.iteritems():
    for i in xrange(len(pics)):
        k = np.random.randint(0, num)
        if k == name:
            if k != 0:
                k -= 1
            else:
                k += 1
        others = images_dic[k]
        other = others[np.random.randint(0, len(others))]
        images_list.append([pics[i], name,
                            pics[i - 1], name,
                            other, k])

print("len of images_list", len(images_list))

np.random.seed(1337)
np.random.shuffle(images_list)
write_levelDB("data_train_leveldb/", images_list[2000:])
write_levelDB("data_test_leveldb/", images_list[0:2000])

```


#### 2.2、模型配置
在模型配置里，我们可以直接使用alex模型或者googlenet模型，他们提供了`train_val.prototxt`文件，这个文件主要用于配置训练模型，可以自定义层数以及每层的参数。尤其是对于卷积层的里参数，需要对CNN有一定的理解。这里不细说CNN模型，只考虑应用。在应用层面，需要注意的是数据层。在数据定义层，Caffe生成的数据分为2种格式：Lmdb和Leveldb。它们都是键/值对（Key/Value Pair）嵌入式数据库管理系统编程库。虽然lmdb的内存消耗是leveldb的1.1倍，但是lmdb的速度比leveldb快10%至15%，更重要的是lmdb允许多种训练模型同时读取同一组数据集。需要注意的一些参数如下：

``` python
layers {
  name: "data"
  type: DATA
  top: "data"
  top: "label"
  data_param {
    source: "facenet/face_train_lmdb"           # 训练数据路径
    backend: LMDB                               # 值得数据格式
    batch_size: 128                             # batch数一般设置为8的倍数，训练比较快
  }
  # 数据变换层
  transform_param {
    crop_size: 227                              # 数据cropsize是模型的真正输入大小
    mean_file: "facenet/face_mean.binaryproto"  # 均值数据路径
    mirror: true
  }
  include: { phase: TRAIN }
}

```

这里需要注意的另一个事情是，在分类层那里（在alex模型里，是fc8层的INNER_PRODUCT里的num_output），需要把默认的类别数改为你自己数据的训练类别数。

#### 2.3、模型训练
如果对都没有啥问题，就可以训练模型了，使用梯度下降法。训练模型之前，我们需要定义solver文件，即`solver.prototxt`，在该文件里，指定迭代次数，是否使用GPU，以及保存中间模型的间隔次数、测试间隔等等。

``` python
net: "facenet/train_val.prototxt"     #指定训练模型配置文件
test_iter: 1000
test_interval: 1000                   # 每迭代1000次，进行一次测试（测试不要太频繁）
base_lr: 0.01                         # 初始学习速率
lr_policy: "step"                     # 学习速率更新方式，每隔多少步更新，也可以使用poly或者constant等等方式
gamma: 0.1                            # 学习速度衰减系数
stepsize: 100000                      # 每迭代这个么次，新学习速率=学习速度乘以衰减系数gamma
display: 2000                         # 每隔两千次打印一次结果
max_iter: 350000                      # 训练一共迭代次数
momentum: 0.9                         # momentum系数
snapshot_prefix: "facenet/"           # 保持中间模型路径
```

之后在caffe目录下，使用imagenet模型提供的`train.sh`。这里建议把各个sh文件和训练数据以及均值文件放一起，配置文件和中间模型放在同一路径，置于sh文件下的子文件，这里可以很容易的知道一个模型是结果是采用了什么配置，避免混乱。train里的命令如下：

`./build/tools/caffe train  --solver=facenet/solver.prototxt -gpu 0  `

这里需要注意，我们可以指定gpu的id（如果存在多个GPU，可以指定具体的GPU）。另外一点，如果我们增加了数据，需要重新训练模型的话，我们可以在训练的时候，指定已训练好的模型来初始化新训练的模型，这样能够加快我们的训练速度。比如

``` python
./build/tools/caffe train  --solver=facenet/solver.prototxt -gpu 0  -weights facenet/caffe_450000.caffemodel

```

那么新训练的模型，不会随机初始化权重，而是更具已训练的caffe_450000.caffemodel来初始化参数。这个初始化参数需要注意，这两个模型是相同的，只是输入数据量增多了。

如果我们finetuning的模型与已经训练不同怎么办呢？比如最开始我训练的模型是150像素大小的，而现在想训练一个250像素的模型，那么我们需要修改新模型的训练配置文件，把数据层的名字更新一下，使得新模型和旧模型的名字不一样，之后指定weight就可以，它会默认根据caffe层相同的名字来使用旧模型来初始化新模型，但是必须保证参数是对应的。如果相同名字的层的参数个数不对应，会报错！

此外，我们训练的方式也有很多种。比如一开始用一个模型训练之后，新增的数据，我们可以合并已有的数据，重新训练新模型，也可以使用旧的模型进行finetuning。在caffe目录下，有一个example和model文件夹，里面有很多例子可以使用，在对应的例子下有readme文件，可以在细节上深入理解。比如我们要输入一对数据，这样的模型如何训练呢？只需要更新一下训练配置文件，可以参考`examples/siamese`下的例子。

### 3、模型使用

模型的使用方式，这里根据caffe提供的python接口来简单介绍一下，这些例子在python文件下，已经提供了一些包装好的接口。而且在example下提供了一些ipython notebook详细的介绍了各个模块的使用。这里需要注意，我们的`deploy.prototxt`文件里，开始有四个input_dim,第一个input_dim是指图片数。由于这个接口是参考了AlexNet模型，所以在python的classify类里有一个系数oversample，默认是True的，意味着在预测的时候会对原始图像crop，默认是crop10张图片。注意在官方给的例子里，是使用了crop，所以他的input_dim是10，正好对应一张照片，他的第四个通道是对应第四个crop图片的特征。一般我们会选择False。此外，需要注意input_dim也对应了net.blobs['blob_name']的照片数量维度。在`net.predict([input_image])`的时候，input_image可以是多张。如果输入张数和input_dim不一致，那么得到的net.blob里的特征数是与input_dim一致的，使得得到的特征与输入的特征无法一一对应。所以建议设置input_dim=2,一次输入两张照片，得到对应的两张照片的特征，用于比对。

此外，除了python的接口，caffe也提供命令行用于特征提取，这些都可以参考官方文档。

### 4、其他
因为caffe的社区非常的强大，多数情况下，你遇到的问题，别人都遇到过了。所以，要善用google。也可以在github或者google邮件列表里，寻找一些答案。