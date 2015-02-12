---
layout: post
title: Linear Models for Classification-Bayesian Logistic Regression
category: 机器学习
tags: [机器学习, PRML]
description: 概述一下线性分类相关的一些知识, 本文主要是贝叶斯Losgistic回归等，使用Laplace平滑方法
---

本篇主要是概述一下线性分类相关的一些知识。分类的目标是在给到一个D维的输入数据X，预测它的类别c（假设共K个类别，一般而言，各个类别是互斥的）。输入空间可以被分割为不同的决策区域(decision regions),这些区域的平面称之为决策边界(decision boundaries或decision surfaces，本文使用决策平面)。这一章，主要考虑线性分类器，即决策平面是输入x的线性模型。如果数据可以完全被线性决策平面分割，称之为线性可分（linearly separable）。笔记分三部分，这是第三部分：以贝叶斯logistic回归为主部分。

<!-- more -->

注：本文仅属于个人学习记录而已！参考Chris Bishop所著[Pattern Recognition and Machine Learning(PRML)](http://research.microsoft.com/en-us/um/people/cmbishop/PRML/)以及由Elise Arnaud, Radu Horaud, Herve Jegou, Jakob Verbeek等人所组织的Reading Group。

###目录
{:.no_toc}

* 目录
{:toc}


###1.拉普拉斯近似
在分类问题中使用贝叶斯法则，比在回归中要复杂的多。我们没办法基于w达到形式的一致化，因为后验概率不再是高斯分布，因此要引入一些基于分析计算和数值抽样的近似方法。这里介绍其中一个方法，拉普拉斯近似(the Laplace approximation)。

拉普拉斯近似的目的是为了找到一个高斯分布来近似一系列连续值的概率密度。首先考虑单变量z，假设其分布为$$p(z) = \frac{1}{Z} f(z)$$，这里$$Z = \int f(z)$$是归一化因子，是未知的。拉普拉斯法的目标就是寻找一个高斯近似q(z)，其中心为p(z)的众数。

我们第一步是得到p(z)的众数，即点$$z_0$$使得$$p'(z_0) = 0$$，等价于$$\frac{df(z)}{dz} \mid_{z=z_0} = 0$$。我们知道高斯分布的对数是变量的二次型，因此可以考虑使用泰勒公式，在ln(f(z))的$$z_0$$处展开，即：

$$\ln f(z) \cong \ln f(z_0) - \frac{1}{2}A(z - z_0)^2; \quad A = - \frac{d^2}{dz^2} \ln f(z) \mid_{z=z_0}$$

这里注意一阶导数为0。那么我们再去指数，可以得到$$f(z) \cong f(z_0) exp(-frac{A}{2}(z - z_0)^2)$$，那么我们可以对于高斯分布得到$$q(z) = (\frac{A}{2 \pi})^{1/2} exp(-frac{A}{2}(z - z_0)^2)$$。这里注意对于高斯分布而言，精度是大于0的，即A > 0这种近似如下图所示。

<img src="/images/prml/ch4_bayesian_laplace.jpg" height="100%" width="100%">

那么我们扩展到M维度，可以知道：

$$\ln f(z) \cong \ln f(z_0) - \frac{a}{2}(z - z_0)^T A (z - z_0)$$

这里Hessian矩阵A是M维度方阵，$$A = \nabla \nabla \ln f(z) \mid_{z=z_0}$$，取指数得到：

$$f(z) \cong f(z_0) exp(-\frac{1}{2} (z - z_0)^T A (z - z_0))$$

那么对应可以得到:

$$q(z) = \frac{\mid A \mid^{1/2}}{(2 \pi)^{M / 2}} exp(-\frac{1}{2}(z - z_0)^T A (z - z_0)) = N(z \mid z_0, A^{-1})$$

为了使得拉普拉斯近似，我们首先要得到$$z_0$$，之后估计在该点的Hessian矩阵。但是实际分布可能是多峰分布，因此也会有不同的拉普拉斯近似。此外，真实分布的归一化因子Z在使用拉普拉斯近似法时是不需要知道的。根据中心极限定理，随着观测量的增加，后验概率会越来越趋近于拉普拉斯估计的高斯分布。拉普拉斯近似的不足是：因为它基于高斯分布，只能用于实数变量。其他的情况，我们也许可以通过一些变换来使用拉普拉斯近似。比如对于 $$0 \le \tau < \infty$$，可以通过对数变换来使用拉普拉斯估计。当然，拉普拉斯近似最严重的不足是它是纯粹依赖于真实分布在变量取得某个特定值，因此可能无法刻画一些重要的全局特征。在后面的章节会有一些别的方法。

###2.模型比较
近似得到p(z)之后，我们可以通过积分得到归一化因子$$Z = \int f(z) dz \cong f(z_0)\frac{(2 \pi)^{M / 2}}{\mid A \mid^{1/2}}$$。

在之前的贝叶斯模型比较中，我们知道模型的evidence是真正重要的，而我们可以根据Z式子近似得到模型的evidence。对于数据集D，有一系列模型$$(M_i)$$，对应参数$$(\theta_i)$$，每一个模型的似然函数是$$p(D \mid \theta_i, M_i)$$。如果我们知道了先验概率$$p(\theta_i \mid M_i)$$，那就很容易得到后验概率。根据$$f(\theta) = p(D \mid \theta)p(\theta); Z = p(D)$$，可以得到$$\ln p(D) \cong \ln p(D \mid \theta_{MAP}) + \ln p(\theta_{MAP}) + \frac{M}{2} \ln (2 \pi) - \frac{1}{2} \mid A \mid$$

这里A是hessian矩阵。上式第一项是在最优参数下的对数概率估计，剩余的三项是奥卡姆因子(Occam factor)，用于惩罚模型的复杂度。如果假设参数的先验概率是高斯分布，而且hessian矩阵是满秩的，那么我们可以近似上式为$$\ln p(D) \cong p(D \mid \theta_{MAP}) - \frac{1}{2}M \ln N$$。这里N是样本数，M是参数个数，该式子也称之为贝叶斯信息准则(BIC, Bayesian Information Criterion)。与第一章的AIC($$\ln p(D \mid w_{ML}) - M$$)相比,对模型的复杂度惩罚更厉害。

复杂度测量，比如AIC、BIC等等，虽然很容易计算，但是也可能导致一些错误的结果。尤其是Hessian矩阵很难满足满秩的假设，我们也可以用拉普拉斯近似来得到一些结果。

###3.贝叶斯logistic回归
用贝叶斯来解释logistic回归是比较棘手的。这里我们采用拉普拉斯近似来解决贝叶斯logistic回归的相关问题。

回想一下拉普拉斯近似，首先要得到后验分布的众数作为高斯分布的中心，那么我们从先验开始。选择高斯分布作为先验分布，即$$p(w) = N(w \mid m_0, S_0)$$，这里$$m_0,S_0$$都是超参数。那么w的后验分布有 $$p(w \mit t) \propto p(w) p(t \mid w)$$，取对数，并化简可以得到：

$$\ln p(w \mid t) = -\frac{1}{2}(w - m_0)^T S_0^{-1} (w - m_0) + \sum^N_{n=1}(t_n \ln y_n + (1-t_n) \ln (1-y_n)) + const$$

这里$$y_n = \sigma(w^T \Phi_n)$$。为了得到后验概率的高斯近似，我们首先最大化后验概率，得到$$w_{MAP}$$，即确定了高斯分布的均值。而方差等于对数似然函数的二阶导数。即：

$$S_N = \nabla \nabla \ln p(w \mid t) = S_0^{-1} + \sum^N_{n=1} y_n(1 - y_n) \Phi_n \Phi_N^T$$

近似的高斯分布形式就是$$q(w) = N(w \mid w_{MAP}, S_N)$$

那么，给定特征向量$$\Phi(x)$$，预测类别$$C_1$$的分布，即$$p(C_1 \mid \Phi, t) = \int p(C_1 \mid \Phi, w) p(w \mid t) dw \cong \int sigma(w^T \Phi) q(w) dw$$，对于二元分类，$$p(C_2 \mid \Phi, t)= 1 - p(C_1 \Phi, t)$$。这里取$$a = w^T \Phi$$，最终得到$$p(C_1 \mid \Phi, t) = \sigma(k(\sigma^2_a) \mu_a)$$，其中$$\mu_a = E(a) = \int q(w) w^T \Phi dw = w^T_{MAP} \Phi$$，而$$\sigma^2_a = var(a) = \int q(w)((w^T \Phi)^2 - (m_N^T \Phi)^2) dw = \Phi S_N \Phi$$， $$k(\sigma^2) = (1 + \pi \sigma^2 / 8)^{-1/2}$$

可以看到，在$$\mu_a = 0$$时，决策边界为$$p(C_1 \mid \Phi, t) = 0.5$$，与使用MAP得到w的情况是一致的。即如果先验概率是均匀分布，又以最小化分类误差为决策准则，那么w的边际分布就没有影响。然而，对于复杂的决策准则而言，w的边际分布起着非常重要的作用。