---
layout: post
title: Probability Distribution
category: PRML
tags: [共轭先验, 概率分布, 混合高斯分布, 狄利克雷分布]
---

该部分主要是复习一下机器学习概率分布的相关知识。这里的概率分布主要讲两个方面：其一是密度估计(Density Estimation)，主要是频率学派和贝叶斯学派的方法。其次是共轭先验，主要是方便后验概率计算。

<!-- more -->

注：本文仅属于个人学习记录而已！参考Chris Bishop所著[Pattern Recognition and Machine Learning(PRML)](http://research.microsoft.com/en-us/um/people/cmbishop/PRML/)以及由Elise Arnaud, Radu Horaud, Herve Jegou, Jakob Verbeek等人所组织的Reading Group。

###内容
**未完待续！！**

###1.二元变量
首先是对于二元变量(Binary Variables)， $$x \in {0,1}$$ ,参数 $$\mu$$ , $$p(x)$$ 满足 $$p(x=1｜\mu)=\mu, p(x=0｜\mu=1-\mu)$$ ，那么我们称$$p(x)$$是伯努利分布(Bernoulli distribution)，表示如下：
$$Bern(x｜\mu) = \mu^x(1-\mu)^{1-x}$$

那么我们如何估计参数$$\mu$$呢？首先是使用频率学派(Frequentist’s Way)的方法，即采用最大似然估计(maximum likelihood estimate)，假设各次观测独立的，我们可以得到：$$\mu^{ML} = \frac{m}{N}$$ ,这里m是观测值中$$x=1$$出现的次数。这种观点很容易导致过拟合(overfitting)，尤其是在观测数N比较小的时候，比如独立投掷硬币3次，均出现正面向上，即 N=m=3，那么 $$\mu^{ML} = 1$$，很可能不合符实际常识。

我们考虑采用贝叶斯学派(Bayesian Way)的方法。贝叶斯学派关注事情是如何发生的，而不仅仅是发生的结果。在使用贝叶斯方法之前，先介绍两个分布。对于二元变量(Binary Variables) $$x \in {0,1}$$ ，这里采用了二项分布(binomial distribution describes)，即描述包含N个观测值的数据集中，$$x=1$$观测值出现次数m的分布，即：
$$Bin(m｜N,\mu) = {N \choose m} \mu^m(1-\mu)^{N-m}$$，其中 $${N \choose m} = \frac{N!}{(N-m)!m!}$$

为了方便计算，选择先验分布为Beta分布，
$$Beta(\mu｜a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$，其中$$\Gamma(x) = \int_0^\infty \mu^{x-1}e^{-\mu}d\mu$$

Gamma函数是实数阶乘的拓展，即$$\Gamma(n) = (n-1)!$$，参数a和b控制着概率密度函数的形状，两者都必须是正值，在这里称之为超参数(hyperparameters), 控制着参数$$\mu$$的分布。对于Beta分布的均值和方差分布如下：
$$E(\mu) = \frac{a}{b} ; var[\mu] = \frac{ab}{(a+b)^2(a+b+1)}$$

根据贝叶斯公式，我们可以得到$$p(\mu｜m,l,a,b) ∝ Bin(m,l｜\mu)Beta(\mu｜a, b)=\mu^{m+a-1}(1-\mu)^{l+b-1}$$, 这里l = N - m。最终得到：
$$p(\mu｜m,l,a,b) = \frac{\Gamma(m+a+l+b)}{\Gamma(m+a)\Gamma(l+b)}\mu^{m+a-1}(1-\mu)^{l+b-1}$$

从这里可以看出：x=1和x=0的实际观测数是m，l，而得到的后验概率中a增加了m，b增加了l，可以简单的认为超参数a和b是有效观测次数。在观测数据相互独立性的假设下，随着观测数据的不断增加（sequential approach），不断的修正着超参数a和b，即实际观测不断的修正着先验认知。

在已经观测数据集D上，如果新来了一个数据x，那如何预测呢？即判断x=1的概率。
$$p(x=1｜D) = \int_0^1 p(x=1｜\mu)p(\mu｜D)d\mu = \int_0^1 \mu p(\mu｜D)d\mu = E(\mu｜D) = \frac{m+a}{m+a+l+b}$$
我们注意到当观测值ml不断增大到可以忽略ab时，参数均值会收敛到最大似然的结果。即大量数据下，贝叶斯和频率学派的结果是一致的。同时也能看出了，观测数据不断上升时，参数的方差也在减小。

拓宽来说，对于普通的贝叶斯推断问题，参数$$\theta$$，考虑到条件期望和条件方差，我们可以推导：

$$\begin{align} E_D[E_\theta[\theta｜D]] &= \int{(\int{\theta p(\theta｜D) d\theta}) p(D)}dD \\ &= \int{(\int{p(\theta｜D) p(D) dD})\theta d\theta} \\ &= \int{p(\theta)\theta d\theta} \\ &= E_\theta(\theta) \end{align}$$

$$\begin{align} var_\theta[\theta] &= E_\theta[\theta^2] - [E_\theta[\theta]]^2 \\ &= E_D[E_\theta[\theta^2｜D]] - [E_D[E_\theta[\theta｜D]]]^2 \\ &= E_D[E_\theta[\theta^2｜D]] - E_D[E_\theta[\theta｜D]^2] + E_D[E_\theta[\theta｜D]^2] - [E_D[E_\theta[\theta｜D]]]^2 \\ &= E_D[var_\theta[\theta｜D]] + var_D[E_\theta[\theta｜D]] \end{align}$$

由此看出参数后验概率的方差是小于先验概率方差的，即参数的不确定性随着观测数据而降低。当然，这只是理想情况，实际过程中，也存在大于的情况。

###2.多元变量
Multinomial Variables


