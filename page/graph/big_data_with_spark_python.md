---
layout: single
category: 知识图谱
tags: 大数据, 课程
description: 大数据spark相关知识汇总和整理，接口使用python的pyspark
title: 大数据spark入门学习
---

本系列以edX的公开课[BerkeleyX: CS100.1x Introduction to Big Data with Apache Spark](https://courses.edx.org/courses/BerkeleyX/CS100.1x/1T2015/info)为基础，汇总和梳理大数据下spark的使用和数据科学的学习。

- update 2015-10-22：
    注意：本文所有课件和作业以及答案的链接均失效（因为我没有把这些东西上传到git上，而且公开课的要求是不能传播答案的。）

<!-- more -->

###目录
{:.no_toc}

* 目录
{:toc}

<br/>

###第一周：入门简介

第一周第一节的[课件](/static/lectures/big_data_Spark/Week1Lec1.pdf)主要简介了课程的一些相关信息以及大数据和数据科学的一些基本模块。课程的主要目标如下：

- 学习数据科学，包括数据获取、清洗、分析和展示等等，尤其是数据产品
- 学习如何展示数据科学，包括理解数据质量、使用机器学习和大数据应用等主题
- 学习编写spark程序并调试程序，包括使用python以及mllib等库

第一周第二节的[课件](/static/lectures/big_data_Spark/Week1Lec2.pdf)主要介绍了数据科学是什么，是多知识的领域结合，包括计算机技术、数学和统计知识以及业务能力。很多人容易陷入Danger Zone,就是缺少数学统计知识，这样会导致很多不靠谱的分析，会犯很多统计学错误，比如伪相关、相关因果等等。之后对比了数据科学和其他科学方向，如下：

- 对比数据库，包括数据价值、数据量、结构化等等。其中，数据库查询过去，数据科学查询未来
- 对比科学计算，包括模型模式、问题结构化、精度等等
- 对比机器学习，机器学习过于学术，而数据科学更关注价值和行动

之后分享了不同的人是如何做数据科学的方法。最终回到数据科学的几个话题上来：
数据获取(Data Acquisition), 数据准备(Data Preparation),分析(Analysis), 数据展示(Data Presentation), 数据产品(Data Products), 观察和实验(Observation and Experimentation)。对于数据科学要克服的问题，记录如下：

- Overcoming assumptions 
- Making ad-hoc explanations of data patterns 
- Not checking enough (validate models, data pipeline integrity, etc.) 
- Overgeneralizing 
- Communication 
- Using statistical tests correctly 
- Prototype ! Production transitions 
- Data pipeline complexity (who do you ask?) 

> We all naturally try to generalize when we see what appears to be a pattern. Also, once we observe a pattern, we tend to see that pattern in future observations. As data scientists we have to try and overcome these tendencies.

之后分别从数据科学的几个话题上介绍了各个模块，尤其是数据的获取与准备，需要很强的ETL(extract, transform, load)能力。也提到了数据在传输过程中的序列化和反序列化，这一点需要注意。课程也从不同的角色（分析员、程序员、企业等等）来看不同的角色在这几个模块使用的工具和方法。同时，总结了一些困难和问题，这些问题是我们做数据科学时需要考虑的，比如多种工具和编程语言使得分享很难，很多分析是一次性的，查找一些脚本往往比自己写还困难。（可能需要搭建知识库！）

###第二周：spark入门

第二周第一个[课件](/static/lectures/big_data_Spark/Week2Lec3.pdf)主要议题如下：

- The Big Data Problem 
    - 传统分析都是基于单机的，但是无法处理大数据，必须使用分布式处理。

- Hardware for Big Data 
    - 硬件也比较便宜，但是容易损坏，有网络传输延时以及性能不均衡等问题。

- Distributing Work 
    - 使用集群计算，以及如何分发任务
    - 简单的介绍了map reduce的思路，使用key-value结构

- Handling Failures and Slow Machines 
    - 分发任务，数据传输非常耗时
    - MR中，机器损坏，可以转移任务
    - MR中，较慢的任务，能够继续分发
    - MR中，不能自动的并行算法

- Map Reduce and Complex Jobs 
    - 复杂的任务需要更多的磁盘IO，非常耗时

- Apache Spark 
    - 内存便宜，传输分享特别快
    - Resilient Distributed Datasets (RDDs)
        - 是一个容错的、并行的数据结构，可以让用户显式地将数据存储到磁盘和内存中，并能控制数据的分区
        - 提供了一组丰富的操作来操作这些数据。比如map、flatMap、filter等转换(transformations)和如join、groupBy、reduceByKey等更为方便的操作（actions，以支持常见的数据运算。
        - 能够自动的从机器损坏中恢复
        - 参考[云笔记-理解Spark的核心RDD](http://note.youdao.com/share/?id=068ff47b94b98db5e1c0a23bd4bef27f&type=note)

    - 对比了Spark和MP的区别



第二周第二个[课件](/static/lectures/big_data_Spark/Week2Lec4.pdf)介绍了spark的编程入门，主要涉及以下几个内容：

- Programming Spark 
    - 使用pyspark接口
    - spark编程分两步：driver program 和 workers program。RDD分布在各个workers里
    - 参考教程，先创建SparkContext，再用它创建RDD 

- RDDs 
    - 主要指pyspark里的RDD，可以通过python的list或者hdfs等其他数据存储构建
    - 参数partitions： more partitions = more parallelism
    - 主要包括transformations和actions操作，其中transformations操作是lazy的，不是即时执行，而是在有action的时候才会执行具体的transformations。
    - 参考官网的[编程入门](http://spark.apache.org/docs/latest/programming-guide.html)和[pyspark API](http://spark.apache.org/docs/latest/api/python/index.html)
    - 例子：创建rDD： `rDD  = sc.parallelize([1, 2, 3, 4, 5],  4)`

- Spark Transformations
    - 主要是从已有的rdd中创建新的rdd
    - 主要操作有map(func), filter(func), distinct([numTasks]), flatmap(func)
    - 注意活用lambda来表达func

- Spark Actions
    - 执行一系列transfos操作，从而获得一些可以不依赖spark的结果
    - 主要操作有reduce(func), take, collect, takeOrdered(n,  key=func)

- Spark Programming Model 
    - 这里主要是讲解了一些操作的内部机理。编程的lifecycle如下
        - 构建RDD
        - 通过lazily transformations转化为新的RDD
        - 选择cache部分RDD
        - 使用一些actions来并行计算并得到结果
    - 注意什么时候使用cache来避免重复读取和重复计算
    - 提及了Key-Value模式下的transformations，主要是 reduceByKey(func),sortByKey(), groupByKey()。
    - pyspark的Closures的介绍，负责任务分发，从而引出pyspark里的两类共享变量Broadcast Variables和Accumulators

本周有一个spark的python notebook的[tutorial](/static/lectures/big_data_Spark/lab/spark_tutorial_student.html)以及[练习](/static/lectures/big_data_Spark/lab/lab1_word_count_student.html)

###第三周：spark结构化数据
第三周第一个[课件](/static/lectures/big_data_Spark/Week3Lec5.pdf)，主要介绍了半结构化数据，列表格数据(Tabular Data)和log日志。

- 数据管理
    - data model： 用于数据描述的概念的集合
    - schema： 使用给定的数据模型来描述一个收集到的特定数据
    - [数据管理](https://en.wikipedia.org/wiki/Data_management)的发展，目前大数据时代采用了file的形式，即分布式文件系统。
    - 概述了一些文件系统的知识

- 半结构化Tabular数据
    - table数据类型的概述
    - 应用广泛，但是存在很多问题，比如定义不明确，数据类型不一致或者一些支持的格式
    - 多数据源的合并，各种数据缺失，数据不一致等
    - pandas使用DataFrame数据格式，可以和pySpark很好的接口和融合。

- 半结构化日志
    - 简单的概述了一些日志的格式以及一些日志分析的常见问题，主要集中在“描述分析”
    - 对于文本数据操作的效率，进行了对比
    > Python performance depends on library you use
    - 文件IO总结，未压缩的情况，读取和写入时间差不多，但是压缩后读取的速度比写入要快很多。使用LZ4压缩比较好 

第三周第二个[课件](/static/lectures/big_data_Spark/Week3Lec6.pdf)，主要介绍了结构化数据和关系型数据库以及SQL在spark里的使用。

- 关系型数据库
    - 关系型数据库，存储一堆关系。relations主要包括schema（每一列的name和type）和Instance。
    - 关系型数据库是最常用的 data model，每一个关系放在一张table里，对应一个schema
    - 概述了一下什么是数据库以及关系型数据库的优缺点，尤其是对稀疏性数据比较浪费

- SQL
    - 通过pySpark DataFrames(SparkSQL)来支持，比较简单的用法，这里不再展示
    - 主要介绍了join的查询以及内在机理。此外有一些比较特别的join模式，INNER JION Enrolled（匹配左侧，且仅显示匹配上的结果），LEFT OUTER JOIN(匹配左侧，未匹配上以NULL显示结果)， RIGHT OUTER JOIN恰好相反。
    - 对于RDDs和pySpark支持 ` inner join(), leftOuterJoin(), rightOuterJoin(), fullOuterJoin()`，需要注意的是pair RDD joins，例如：

{% highlight sh %} 

>>> x = sc.parallelize([("a",   1), ("b",   4)])    
>>> y = sc.parallelize([("a",   2), ("a",   3)])    
>>> sorted(x.join(y).collect()) 
Value:  [('a',  (1, 2)),    ('a',   (1, 3))]

{% endhighlight %}

最后是本周的[作业](/static/lectures/big_data_Spark/lab/lab2_apache_log_student.html)，主要是log日志分析，没用到sql，还是跟上周差不多的命令。

###第四周：spark数据分析与mllib
第四周第一个[课件](/static/lectures/big_data_Spark/Week4Lec7.pdf)，主要介绍了数据清洗的一些认识。

- 数据清洗(data cleaning)
    - 分别从统计观点、数据库、业务以及数据科学的角度解读数据清洗
- 数据质量(data quality)
- 数据聚合(data gathering)
    - What are you doing with all this data anyway?
- 数据质量控制和度量(data quality constraints and metrics)
    - 常用技术如下
        - 过程管理
        - 统计分析
        - 数据库关系约束
        - 业务知识理解
- 数据集成(data integration)
    - 数据集成过程中伴随着很多数据质量问题，需要注意

第四周第一个[课件](/static/lectures/big_data_Spark/Week4Lec8.pdf)，主要介绍了基于spark的数据分析探索和机器学习库mllib。

- 数据分析(Exploratory Data Analysis)
    - 描述性（descriptive）分析和推断性（inferential）分析
    - 举例一些常见的商业问题：包括基础统计、假设检验、分类、预测等等问题，以及这些问题对应的解决方法。
- 重要的概率分布（Some Important Distributions）
    - Understand your data’s distribution before applying any model
- 机器学习库mllib
    - mllib库是spark下的分布式机器学习库，实现了许多分类、回归和聚类等模型，以及一些优化算法。
    - 可以参考spark的[mllib文档](http://spark.apache.org/docs/latest/mllib-guide.html)
    - lab里介绍了协同过滤模型(Collaborative Filtering)。协同过滤的实现方式比较多，这里采用了ALS(Alternating Least Squares）算法来实现。
    - 更加详细的教程可以参考新的课程[Scalable ML BerkeleyX MOOC](https://courses.edx.org/courses/BerkeleyX/CS190.1x/1T2015/info)

最后是本周的[作业](/static/lectures/big_data_Spark/lab/lab3_text_analysis_and_entity_resolution_student.html)，主要是文本相似度分析，基于BOW。这类问题，主要在于确定基本思路，之后实现模型中的各个点，再整合起来就可以了。

###第五周：协同过滤lab
本周只有[lab](/static/lectures/big_data_Spark/lab/lab4_machine_learning_student.html)，完成协同过滤模型做推荐系统。使用mllib库的主要问题在于，如何组织你的数据，使其符合mllib的数据格式，并且学会评估模型的好坏，学会选择模型和调整模型。