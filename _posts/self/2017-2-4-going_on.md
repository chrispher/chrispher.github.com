---
layout: post
title: 连接2016与2017年
category: 无伤大雅
tags: [个人观点]
description: 总结与展望
---

2016年的博客,看似写的少,却是比往年更加的认真, 深入和专注. 2017年,会陆续做一些放出和连续性的教程类文章.

<!-- more -->

### 1.2016年的博客更新与知识管理
有朋友跟我说我2016年下半年都不怎么更新博客了. 其实确实是这样的, 因为读了一些论文,把这些论文再整理成文章,确实很麻烦. 而且很多文章, 比如残差网络, GAN, RNN等比较时髦的模型, 很多人都对论文做过一些概述和翻译, 尤其是**机器之心**, 跟进新论文确实蛮快的. 所以, 我在自己的papers下,对于很多论文都是做了一些基本的标注和解释就完了, 没有写成博客. 下图是我个人的papers管理:

![show png](/images/resource/personal_papers2.png)

对于某篇具体的论文,一般都是简单的写写注释:

![show png](/images/resource/personal_papers.png)

当然啦,这些是读论文. 2016年的学习中,最大的成长不是阅读了多少论文,或者跟着一些公开课(比如cs 224d等)复习或学习了一些知识, 而是在16年做了不少的coding训练. 这一方面是因为工作中需要coding的部分, 另一方面也是自己私下写一些c++或者python库用于更好的理解模型. 这些都在我自己的一个知识库里做了保留, 比如下图是我个一个git库, 其中chrispher里是笔记, 这个笔记和博客的笔记不一样, 是类似于[tsne完整版](http://www.datakit.cn/blog/2017/02/05/t_sne_full.html)这种的笔记, datasets下是一个脚本, 用于下载一些公开的数据集,来做一些练习. dwarf是一个c++的神经网络库,这个写的比较简单,仿的caffe,用的eigen库和protobuf, 主要是练手c++吧. hobbit是在16年初设计的kitnet上的基础之上做的重构和优化,是我个人的模型原理库,涉及到决策树,神经网络,t-sne等等模型的源码, 神经网络那部分是完全的自己设计和手写,t-sne等代码有部分参考了论文作者的一些实现. 在写这个hobbit的时候,让我对很多模型和优化等等有了更深入的认识. 另外project文件夹下,主要就是一些项目练习,比如测试一下CNN的不同结构的结果差异, loss变化等等.这一块在16年做的不够深入.

![show png](/images/resource/personal_notes.png)


总体来说, 2016年做的还算可以, 有成长, 但是没有成长的那么迅速.有一些明显的不足, 比如有些模型的数学公式虽然知道了, 但是并没有真正的体会到其中的意义的价值, 比如t-sne里,如果不是完整的阅读论文的话和深入思考的话,对于里面提到的很多问题都是一知半解,而且更打脸的是作者的很多思考方式对于我而言是全新的, 我之前从未这样的去认识一个问题. 在工作中也是一样,把学习到的东西更好的和业务\场景结合,更好的理解世界, 这一块我做的还是很差的.

### 2.2017年做点什么
做点什么,当然是针对博客啦, 对于个人的17年的计划还在进一步思考中, 好像不应该拖到2月份...17年, 博客方面的话,

- 写一些比较经典的论文笔记, 不是那种热门论文的概述翻译, 会挑一些经典的稍微老一点的论文做一些笔记, 因为这些论文上的方法,我之前都是在书上看到的, 或者一些介绍上, 没有想过原作者是怎么思考的, 这个是tsne这篇文章给我启示吧. 可以去读一读Jordan或者Hinton等写的一些方向性概述的文章, 比如MCMC概述之类的, 这些文章一般都会总结过去,认识现在,看看未来,以及他们是怎么思考这些问题的.
- PRML这本书的笔记做了一半吧,看百度流量的后台,阅读量一直很小,反而是一些扯淡的,比如caffe框架应用那个比较高. 难道大家都很懂原理吗? 开源软件使用,应该是最简单的吧, 看一些官方文档就可以了, 表示不理解. 我自己看了一些自己写的确实有点拗口,不利于理解. 随着我自己的成长, 想用简单的方式来解释和分享,但是确实是比较耗费精力. 所以后续会用比如pytorch或者什么书本之类的作为模板, 写一些入门性质的介绍和动手方面的
- 实战方面,很多trick都是要自己去尝试的,而且kaggle上有很多冠军分享的trick可以学习一下. 有人问工作里怎么用,比如怎么训练模型,怎么部署等等,这些问题等你工作了,到了那个环境了,应该很快就会了.关键还是要懂一点机器学习的基础.
- 当然,我个人的git库是在gitlab上,私人的. 目前并不开放的, 里面的总结是稍微深入的总结, 笔记也会不断的修正. 博客上的东西,太久的就懒得去完善了,比如PRML系列. 而且博客上注重原创,有一些网上整理和引用的东西, 我个人笔记里会有, 但是发在博客上不太合适.

### 3. 最后
看到后台, 不少人看了t-sne, 那么就把个人完整版的tsne放出来[tsne完整笔记](http://www.datakit.cn/blog/2017/02/05/t_sne_full.html), 给大家一起学习下, 从笔记上也能看到我个人私下是如何整理经典论文的.

2017年的成长, 希望不局限在技术/知识层面, 在问题的理解和世界的认知上, 也要多思考!


