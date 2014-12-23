---
layout: post
title: Linear Models for Regression
category: PRML
tags: [线性回归, 基函数, 预测]
---

本篇主要是概述一下线性回归相关的一些知识。在回归模型中，主要目标是给到一个D维的输入数据X，预测出一个或多个的连续目标值t。这里涉及的不仅仅是简单的线性回归模型，包括了一些非线性变化，比如基函数等等。

<!-- more -->

注：本文仅属于个人学习记录而已！参考Chris Bishop所著[Pattern Recognition and Machine Learning(PRML)](http://research.microsoft.com/en-us/um/people/cmbishop/PRML/)以及由Elise Arnaud, Radu Horaud, Herve Jegou, Jakob Verbeek等人所组织的Reading Group。

###目录
- [1.线性回归模型](#1.线性回归模型)
- [2.偏差方差分解](#2.偏差方差分解)

**待更新！！**

<a name="1.线性回归模型"/>

###1.线性回归模型
一个基本的例子是函数拟合，观测到N个D维度的x，每个都都对应一个目标值t，构造一个函数y(x)用于预测t。在概率的观点下，是寻找一个合适的概率分布$$p(t \mid x)$$。如下图所示：

<img src="http://chrispher.github.com/images/prml/ch3_LinearRegression.jpg" height="80%" width="80%">

####1.1基函数
基础的线性回归模型形式是：$$y(x,w) = \sum_{j=0}^{M-1} w_i \phi_j(x) = w^T \phi(x)$$, 这里$$w = (w_0,...,w_{M-1})^T ; \ \ \phi=(\phi_0,...,\phi_{M-1})^T, \phi_0(x)=1$$， $$w_0$$是偏置参数（也就是常说的截距项）。这里的$$\phi(x)$$就是基函数，对x的每个属性都有一个基函数，常用的一些基函数有

- 多项式：$$\phi_j(x) = x^j$$；
- 高斯：$$\phi_j(x) = exp(-\frac{(x-\mu_j)^2}{2s^2})$$
- Sigmoidal：$$\phi_j(x) = \sigma(\frac{x-\mu_j}{s}) \ 其中 \  \sigma(a) = \frac{1}{1+e^{-1}}$$

还有其他，比如傅里叶、小波、样条等等。基函数的一个直观的作用是引入了非线性变换。

####1.2最大似然与最小二乘
我们假设了数据是由函数y(x,w)决定的，即$$t = y(x,w) + \xi$$, 这里$$\xi$$是均值为0，精度为$$\beta$$的高斯分布(高斯噪声),我们可以认为$$p(t \mid x,w,\beta) = N(t \mid y(x,w), \beta^{-1})$$。在IID条件下，可得到如下似然函数：

$$p(t \mid X, w, \beta) = \prod_{n=1}^N N(t_n \mid \underbrace{ w^T \phi(x_n)}_{mean} , \underbrace{ \beta^{-1}}_{var})$$

使用最大似然估计， 得到：

$$\ln p(t \mid w,\beta) = \sum_{n=1}^N \ln N(t_n \mid w^T \phi(x_n), \beta^{-1}) = \frac{N}{2} \ln \beta - \frac{N}{2} \ln(2\pi) - \beta E_D(w)$$

其中，$$E_D(w) = \frac{1}{2} \sum_{n=1}^N (t_n - w^T \phi(x_n))^2$$。对数似然函数对w取导数，可以得到: 

$$\begin{align} \nabla \ln p(t \mid w, \beta) &= \sum_{n=1}^N (t_n - w^T \phi(x_n)) \phi(x_n)^T \\ &= \sum_{n=1}^N t_n \phi(x_n)^T - w^T (\sum_{n=1}^N \phi(x_n) \phi(x_n)^T) \\ &= 0 \end{align}$$

最终得到：$$w_{ML} = (\Phi^T \Phi)^{-1} \Phi^T t$$。这里看到了$$(\Phi^T \Phi)$$这种内积形式，那么可以联想到核方法。此外，$$\beta_{ML}^{-1} = \frac{1}{N} \sum_{n=1}^N (t_n - w^T_{ML} \phi(x_n))^2$$。我们可以用几何解释一下回归分析，最小二乘法就是在特征空间里寻找与目标向量t最接近的向量y，而该向量是目标向量t在特征空间的投影。如果$$(\Phi^T \Phi)$$是奇异的，换句话说就是特征空间中，基向量共线或接近共线，那么$$(\Phi^T \Phi)$$就不存在逆矩阵了，导致参数会有很大的浮动（没有唯一解），可以考虑使用SVD或其他方法求解。如下图所示：

<img src="http://chrispher.github.com/images/prml/ch3_Geometry_of_least_squares.jpg" height="100%" width="100%">

####1.3顺序学习 
顺序学习(Sequential learning)，也称为“在线学习”(on-line),可以使用随机梯度下降(stochastic gradient descent)来实现。即$$E_D(w) = \frac{1}{2} \sum_{n=1}^N (t_n - w^T\phi(x_n))^2$$。在上一部分，我们是直接得到了w的解析解，这里我们使用梯度下降法，选择学习速率$$\eta$$，得到w的更新表达式如下：

$$w^{\tau + 1} = w^{\tau} + \eta \underbrace{(t_n - {w^(tau)}^T \phi(x_n)) \phi(x_n)}_{\nabla E_n}$$

####1.4正则最小二乘
接下来就是引入正则项，正则项的优点之一是能够控制模型的复杂度。在误差函数上增加正则项：$$E_D(w) + \lambda E_W(w)$$, 其中$$\lambda$$是正则系数，控制着基于数据的误差$$E_D(w)$$和正则项$$\lambda E_W(w)$$，比较常见的正则项是$$E_W(w) = \frac{1}{2w^Tw}$$,因此误差函数变为:

$$\frac{1}{2} \sum^N_{n=1} (t_n - w^T \phi(x_n))^2 + \frac{\lambda}{2} w^Tw$$

选择一个合适的正则项，有的称之为“权重衰减”(weight decay),即它使得权重尽可能的小到0；在统计学里，也称之为“参数收缩”(parameter shrinkage)。因为包含了w的二次项，可以直接得到一个闭合解，即另导数为0，得到：

$$w = (\lambda I + \Phi^T \Phi)^(-1) \Phi^T t$$

一个更广泛的正则方法如下：

$$\frac{1}{2} \sum^N_{n=1} (t_n - x^T \phi(x_n))^2 + \frac{\lambda}{2} \sum^M_{j=1} \mid w_j \mid^q$$

对于之前的情况是，q=2。q=1就是统计学里的lasso的方法，当 $$\lambda$$很大时，可以使得$$w_j$$等于0，从而实现稀疏模型(sparse model);为了看到这一点，我们可以把带正则系数的项，可以看成是参数的限制，即$$ \sum^M_(j=1) \mid w_j \mid ^q lq \eta$$，这样对于一个给定的$$\eta$$，可以通过“拉格朗日乘子法”得到我们之前的误差函数。假设只有$$w_1,w_2$$,我们可以看到当q=2的时候，正则项是一个圆，而当q=1的时候，正则项是一个菱形。等高线图是我们误差函数(不带正则项)的在参数空间的投影。交点就是我们最终的$$w$$值，可以看到q=1的时候，$$w_1$$是可以取得0的点，即$$w_1$$对应的属性$$x^(1)$$就没有作用了。

<img src="http://chrispher.github.com/images/prml/ch3_regular_sparse.jpg" height="100%" width="100%">

此外，如果我们的目标t的维度是大于1的呢？其实是不影响的，只有一维一维的计算就可以了，这样w就是M×K维的了。最终结果就是$$W_{ML} = (\Phi^T \Phi)^{-1} \Phi^T T$$

<a name="2,偏差方差分解"/>

###2.偏差方差分解
我们知道过拟合是最大似然的一个不幸的特点，但是通过贝叶斯方法却是可以避免的。在考虑贝叶斯方法之前，我们先讨论模型复杂度的问题，也称之为“偏差-方差的权衡”(biasvariance trade-off)，这是频率学派对于模型复杂度的观点。过拟合一般发生在基函数比较多（即特征多）而训练数据有限的情况下。限制基函数数量，能够限制模型的复杂度。而使用正则项也能够控制模型的复杂度，但是需要决定正则系数是多少。

这里我们深入的理解一下。对于回归问题，loss-function是 $$L(t,y(x)) = (y(x) - t)^2$$，决策问题是最小化误差，即$$E[L] = \int \int (y(x) - t)^2 p(x,t)dx dt$$，最终得到$$y(x) = \int t p(t \mid x) dt = E_t[t \mid x]$$,这就是回归函数。这里需要区分一下有决策理论得到的误差函数和最小二乘法得到的误差函数，虽然这里看起来一样，实际上决策理论得到的可以用正则或贝叶斯方法得到$$p(t \mid x)$$。我们知道生成分布是$$p(t \mid x)$$，那么理论上观测到的数据是$$h(x) = E[t \mid x] = \int tp(t \mid x)dt$$,实际观测到的是t，而我们拟合得到的是y(x)，那么，我们的loss-function可以分解为:

$$E[L] = \int (y(x) - h(x))^2 p(x)dx + \int(h(x) - t)^2 p(x,t)dxdt$$

上式子就是在原式子上增加和减去了h(x)项。理论上可以找到最优的y(x)使得第一项达到0（假设数据和计算能力均无限）。而第二项是数据中的噪声引起的，表示了预期可达到的最小期望损失(expected
squared loss)

对于给定的数据集D，我们的预测函数是$$y(x;D)$$，那么对于上式第一项有：

$$E_D[(y(x;D) - h(x))^2] = \underbrace{E_D[y(x;D) - h(x)^2]}_{(bias)^2} + \underbrace{E_D[(y(x;D) - E_D[y(x;D)])^2]}_{variance}$$

合起来就是 expected loss = (bias)$$^2$$ + variance + noise, 这里(bias)$$^2 = \int (E_D[y(x;D)] - h(x))^2 p(x)dx$$; variance = $$\int E_D[(y(x;D) - E_D(y(x;D)))^2]p(x)dx$$; nois = $$\int (h(x)-t)^2 p(x,t)dxdt$$。我们把期望损失分解成了bias(偏差)、variance(方差)与常数nose(噪声)的和。灵活的(flexible)模型一般都有较低的bias和较高的variance；而死板的(rigid)模型一般对于较高的bias和较低的variance。虽然这个分解让我们看到了模型复杂度内在的一些东西，但是实际应用确实有限的，因为着要求多个数据集。

<a name=""/>

###3.贝叶斯线性回归