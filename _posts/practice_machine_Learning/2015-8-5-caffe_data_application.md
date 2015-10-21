---
layout: post
title: caffe的应用
category: 深度学习
tags: [环境搭建, 机器学习]
description: 关于深度学习框架caffe的使用的一些技巧。
---

本文主要介绍深度学习框架caffe的一些使用心得，主要参考[caffe官方文档](http://caffe.berkeleyvision.org/tutorial/)。都是简单的应用，caffe可以方便快速的做一些项目，但是真正搞深度学习的话，还是要研究代码，自己做一些事情的。

<!-- more -->

###目录
{:.no_toc}

* 目录
{:toc}


###1、caffe模型
caffe的模型是graphic的模式，你需要选择各个模块，定义这些模块的输入和输出，以及参数空间。这样的做法，使得你可以向堆积木一样，构造出类似于inception的模块，以及siamese等等模型。灵活的构建属于自己的深度学习模型结构。

caffe官方的mode zone里，提供很多比较流行的模型和任务，大家可以根据这些项目和模型，调整并得到自己的模型。

###2、FineTuning
如果你的数据量有限，那么，一般不建议自己完全从头训练起caffe模型。一般是找相关的项目或者模型，先finetuning一下，之后再慢慢的调整。一般fine tuning的方式，都是把learning rate（solver.prototxt）调低（为原来的十分之一），之后把训练模型的最后一层或者两层的学习速率调大一点————这就相当于，把模型的前面那些层的学习调低，使得参数更新的慢一点以达到微调的目的。

微调的时候，有时候训练数据特别少，而且希望模型的前面几层的参数保持不变。方法是使得这几个层的学习速率为0就可以了，比如设定lr_mult为0。

###2、多任务与多输入
这里最核心的是理解数据的输入和输出，灵活的应用"Slice","Flatten"等等数据处理模块。比如siamese模块，需要输入两张照片，之后用slice切成两张照片，分别跑一遍模型，之后利用参数的命名，使得两个模型共享同一个参数（可以参考caffe的exampl里的siamese）。

如果是多任务的话，比如输入是一张照片，经过一系列的卷积或者LRN等等模块，得到一系列特征，之后把特征与不同的label放在一起，使用不同的loss即可以。如果是多输入和多输出，方法是一样的。

当然，多loss也是可以，不同的loss可以有不同的权重，可以参照GoogleNet，里面设置了三个loss。

因为我一直是python做数据处理，所以简单的给一个例子，是读取三张照片，作为输入。采用levelDB。

{% highlight python %}
def write_levelDB(dbname, images):
    # dbname = data_train_leveldb/
    db = leveldb.LevelDB(dbname, create_if_missing=True,
                         error_if_exists=True, write_buffer_size=268435456)
    wb = leveldb.WriteBatch()

    for count, pic in enumerate(images):
        img1, label1 = pic[0], pic[1]
        img2, label2 = pic[2], pic[3]
        img3, label3 = pic[4], pic[5]
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

{% endhighlight %}

###4、caffe社区与调参
很多新的算法或者新的模块，caffe社区里实现是很快的，但是caffe的merger进度一般来说比较慢。所以，遇到一些问题，可以先去caffe社区里找找答案。

当然，也有很多人基于caffe自己改版了很多版本出来，这些在github里还是蛮多的，包括triplet等等实现。

关于调参，很多时候都是经验。比较忧桑的是，你在这个情景下，增加一个模块效果是好的，在另一个场景可能就是不好的。没有一个很好的理论支持你，应该怎么调整。只能简单的说一下，学习速度、batch number之类的东西。如果batch number特别小的话，会导致loss不收敛。一般调参的策略是，先把试一试参数，然后把display设的小一点，多看看结果，找到一个合适的参数之后，再整个的跑。

另外，很多论文都提出一些参数初始化以及不同的激活函数等等的，一般都是先使用论文里给出的一些参数和初始化方法，效果通常不差。


###5、keras
最后，推荐一下[keras](https://github.com/fchollet/keras)——一个基于theano的python深度学习库，受torch的启发，非常的模块化，支持 Sequential 和 Graph 的方式，灵活定义自己的模型，同时不需要想caffe那样定义那么长的模型文件（吐槽一下，训练模型的prototxt，用了7K多行！！），更重要的是，我个人比较熟悉python，可以自己写很多东西，包括自定义loss等等。
看了keras的一些规划，以后的底层很可能就脱离theano了，这样可能会更好一点（theano的调试，很坑。。）


