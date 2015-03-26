---
layout: post
title: TAE：文本分析
category: 统计分析
tags: [TAE, 数据分析]
description: The Analytics Edge课程下的文本分析入门。
---

The Analytics Edge系列的四节笔记，这是第四节：文本分析入门。本节课主要讲述了介绍了文本分类的问题，涉及一些自然语言处理的基本概念。

<!-- more -->

系列笔记主要根据the analytics Edge by MITx  课件做的笔记。这门课程，主要通过各个案例来学习数据分析方法。

###目录
{:.no_toc}

* 目录
{:toc}


###课堂简要
在文本分析中，面临的第一个问题是如何获得数据集。在课堂中，主要是采用了亚马逊的众包平台。而在实际业务中，通常都是需要在产品设计之初，就设定一些评分指标，比如好评、差评等，以有利于后期的发展规划。  
这里需要注意文本的预处理，预处理对最终模型的效果影响是非常非常大的。预处理需要注意：大小写、标点符号、**停用词**。此外，主要注意以下四个处理：
- 有些时候对于网址、数字、特殊字符等有助于我们分类目标的文本进行特殊处理，比如在垃圾邮件分类中，会将所有网址链接转化为一个**自定义单词**‘mailhttps’。当然，在某些自然语言处理包中，还会有**拼写纠正**、词性划分等等。
- 对于stemming（词干提取），课程中提到了一些方法，如设计词库、设计规则等方式。
- 稀疏性，对于一些出现次数特别少的单词，比如稀疏性大于99.5%等词语，需要删掉（即在所有文档中出现次数小于一定次数的单词）。
- 单词权重，tf-idf是比较常用的一种处理方式。  

此外，在IBM案例中，提到了沃森的工作方式：分析问题（问题是寻找什么样的答案）——产生假设（寻找所有可能答案）——评估假设（评估各个假设的置信度）——排序结果（提供支持度最高的答案）。

###案例一
需要注意读取文件时的两个参数stringsAsFactors和encoding="latin1"  
- **read the dataset**  
`tweets = read.csv("tweets.csv", stringsAsFactors=FALSE) `  
`str(tweets)`  
`tweets$Negative = as.factor(tweets$Avg <= -1)`  
`table(tweets$Negative) `   
`library(tm)`  
`library(SnowballC)`  
- **Create corpus, Convert to lower-case, Remove punctuation**   
`corpus = Corpus(VectorSource(tweets$Tweet))`   
`corpus = tm_map(corpus, tolower) `  
`corpus = tm_map(corpus, removePunctuation)`  
- **Remove stopwords and apple, Stem document**  
`stopwords("english")[1:10]`  
`corpus = tm_map(corpus, removeWords, c("apple", stopwords("english"))) `  
`corpus = tm_map(corpus, stemDocument) `  
- **frequencies, sparsity**  
`frequencies = DocumentTermMatrix(corpus) `  
`inspect(frequencies[1000:1005,505:515]) `  
`findFreqTerms(frequencies, lowfreq=20)`  
`sparse = removeSparseTerms(frequencies, 0.995) # Remove sparse terms`  
Convert to a data frame  
`tweetsSparse = as.data.frame(as.matrix(sparse))`  
Make all variable names R-friendly and Add dependent variable   
`colnames(tweetsSparse) = make.names(colnames(tweetsSparse))`  
`tweetsSparse$Negative = tweets$Negative`
- **Split the data**  
`library(caTools)`  
`set.seed(123)`  
`split = sample.split(tweetsSparse$Negative, SplitRatio = 0.7)`  
`trainSparse = subset(tweetsSparse, split==TRUE)`  
`testSparse = subset(tweetsSparse, split==FALSE)`  
之后，根据得到的数据，建模即可。
