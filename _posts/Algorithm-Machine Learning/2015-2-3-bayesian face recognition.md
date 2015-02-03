---
layout: post
title: bayesian face recognition
category: Algorithm-Machine Learning
tags: [Bayesian Face, 分类, 预测]
---

本篇文章主要是关于joint bayes在人脸识别中的应用。主要参考论文[bayesian face recognition](http://www.baidu.com/link?url=Gy5Z2WOhAYFyWUauyY_h3wVA0Z5C84lS3WxOmoReHUNRZbEOIPOf-YBWOVLXmPYHD3seJJ7caR58cVeiZQDnhZgZ9YQGl060tosIxfliY7oPF5vi-PDCydSNSe4fLpPC)和[Bayesian Face Revisited: A Joint Formulation](http://home.ustc.edu.cn/~chendong/JointBayesian/)。我们首先看bayesian下的人脸识别，之后再看一下joint bayes。

<!-- more -->


###目录

###1、引入
贝叶斯人脸识别是由Baback Moghaddam等人成功应用到到人脸比对（是否是同一个人）中，把比对作为二元贝叶斯决策问题。这里我们令$$H_I$$表示两张脸$$x_1, x_2$$属于同一个人的假设(intra-personal hypothesis)，令$$H_E$$表示不是同一个人的假设(extra-personal hypothesis )。那分类问题就是判断$$\Delta = x_1 - x_2$$ 是否为同一个人，两张脸之间的相似度为$$S(x_1, x_2) = P(\Delta \in H_I) = P(H_I \ mid \Delta)$$，这里$$P(H_I \ mid \Delta)$$是后验概率。之后，我们会看到该方法是LDA(Linear Discriminant Analysis)的一种非线性扩展。

###2、贝叶斯人脸识别
对相似度(后验概率)使用贝叶斯法则，可以得到：

$$S(x_1, x_2) =  P(H_I \ mid \Delta) = \frac{P(\Delta \mid H_I)P(H_I)}{P(\Delta \mid H_I)P(H_I) + P(\Delta \mid H_E)P(H_E)}$$

这里先验概率P(H)是根据训练数据得到的，也可以自定义指定。这个表达式是针对二元分类的，我们也可以扩展到多元分类。由最大后验概率之后，可以得到在$$P(H_I \mid \Delta) > P(H_E \mid \Delta)$$或者等价于$$S(x_1, x_2) > \frac{1}{2}$$，即可认为是同一个人。如果我们只使用$$S' = P(\Delta \mid H_I)$$，即使用最大似然(ML, maximum likelihood)方法，在多数情况下与MAP的结果差不多(数据量足够的情况下，MAP和ML结果基本一致)。

这里有一个问题，就是维度特别大的时候，样本之间没有充分的独立性来得到有效的$$\Delta$$，即使能够得到这个值，也会因为高维度使得似然概率难以得到。一个解决方法是使用PCA降维到M个主成分对应的空间。这里把原始的N维空间分成一个包含M个主成分的子空间F和其余成分N-M的子空间F',假设$$x_1, x_2$$服从高斯分布，那么有：

$$P'(\Delta \mid H) = frac{exp(-\frac{1}{2} \sum^M_{i=1} \frac{y_i^2}{\lambda_i}){(2 \pi)^{M / 2} \prod^M_{i=1} \lambda_i^{1/2}} frac{exp(-\frac{}{})}{(2 \pi \rho)^{(N - M)/2}} = P_F(\Delta \mid H) P'_{F'}(\Delta \mid H)$$

这里通过最小化交叉熵得到权重参数$$\rho$$为F'空间里的特征值的均值，即$$\rho = \frac{1}{N - M} \sum^N_{i=M+1} \lambda_i$$。 

接下来，我们考虑特征向量$$\Delta = x_j - x_k$$，假设对应的类别条件服从高斯分布，那么有：

$$P(\Delta \mid H_E) = \frac{e^{-frac{1}{2} \Delta^T \Sigma^{-1}_E \Delta}}{(2 \pi)^{D / 2} \mid \Sigma_E \mid ^{1/2}}$$

$$P(\Delta \mid H_I) = \frac{e^{-frac{1}{2} \Delta^T \Sigma^{-1}_I \Delta}}{(2 \pi)^{D / 2} \mid \Sigma_I \mid ^{1/2}}$$

之后，我们通过一些处理使得计算似然概率$$P(\Delta \mid H_I);P(\Delta \mid H_E)$$更加方便。首先是都图片进行“白化”(whitening)处理。我们用i表示同一个人白化的结果，e表示不同的人的白化的结果，那么$$i_j = \Lambda_I^{-\frac{1}{2}} V_I x_j; \ \ e_j = \Lambda_E^{-\frac{1}{2}} V_E x_j$$，这里$$\Lambda; V$$分别表示矩阵的最大特征值和特征向量。在预处理之后，可以得到似然结果如下：

$$P(\Delta \mid H_E) = \frac{e^{-\frac{1}{2} \mid \mid e_j - e_k \mid \mid^2}}{(2 \pi)^{D / 2} \mid \Sigma_E \mid ^{1/2}}$$

$$P(\Delta \mid H_I) = \frac{e^{-\frac{1}{2} \mid \mid i_j - i_k \mid \mid^2}}{(2 \pi)^{D / 2} \mid \Sigma_I \mid ^{1/2}}$$

之后使用贝叶斯法则就可以了。









基于最大后验概率(MAP, Maximum a Posterior)，可以得到一个对数似然比例作为决策，即$$r(x_1, x_2) = \log \frac{P(\Delta \mid H_I)}{P(\Delta \mid H_E)}$$， 该式子也可以看作是衡量人脸比对问题中，$$x_1$$和$$x_2$$直接的相似度。