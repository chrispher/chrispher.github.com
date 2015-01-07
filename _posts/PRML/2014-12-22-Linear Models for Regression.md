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
- [3.贝叶斯线性回归](#3.贝叶斯线性回归)
- [4.贝叶斯模型比较](#4.贝叶斯模型比较)
- [5.Evidence近似](#5.Evidence近似)

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

选择一个合适的正则项，有的称之为“权重衰减”(weight decay),即它使得权重尽可能的小到0；在统计学里，也称之为“参数收缩”(parameter shrinkage)。因为包含了w的二次项，可以直接得到一个闭合解，即令导数为0，得到：

$$w = (\lambda I + \Phi^T \Phi)^{-1} \Phi^T t$$

一个更广泛的正则方法如下：

$$\frac{1}{2} \sum^N_{n=1} (t_n - w^T \phi(x_n))^2 + \frac{\lambda}{2} \sum^M_{j=1} \mid w_j \mid^q$$

对于之前的情况是，q=2。q=1就是统计学里的lasso的方法，当 $$\lambda$$很大时，可以使得$$w_j$$等于0，从而实现稀疏模型(sparse model);为了看到这一点，我们可以把带正则系数的项，可以看成是参数的限制，即$$ \sum^M_{j=1} \mid w_j \mid ^q \le \eta$$，这样对于一个给定的$$\eta$$，可以通过“拉格朗日乘子法”得到我们之前的误差函数。假设只有$$w_1,w_2$$,我们可以看到当q=2的时候，正则项是一个圆，而当q=1的时候，正则项是一个菱形。等高线图是我们误差函数(不带正则项)的在参数空间的投影。交点就是我们最终的$$w$$值，可以看到q=1的时候，$$w_1$$是可以取得0的点，即$$w_1$$对应的属性$$x^{(1)}$$就没有作用了。

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

<a name="3.贝叶斯线性回归"/>

###3.贝叶斯线性回归

####3.1参数分布
假设噪声是精度为$$\beta$$的高斯分布，似然函数$$p(t \mid w)$$是w的二次指数形式，共轭先验选择高斯分布$$p(w) = N(w \mid m_0, S_0)$$, 那么后验分布也是高斯分布。

$$p(w \mid t) = N(w \mid m_N, S_N) \propto p(t \mid w)p(w)$$

这里$$m_N = S_N(S_0^{-1}m_0 + \beta \Phi^T t) \ \ ; \ \ S_N^{-1} = S_0^{-1} + \beta \Phi^T \Phi$$。我们知道高斯分布的概率密度取均值时，最大。那么最大后验概率，就可以令
w等于均值即可，即$$w_{MAP} = m_N$$。同样，这里也可以使用序列学习的框架。

接下来，我们具体的细化一下。为了方便处理，我们现在考虑一个特定形式的先验高斯分布，由超参数$$\alpha$$控制精度，均值为0，即$$p(w \mid \alpha) = N(w \mid 0, \alpha^{-1}I)$$。那么刚才得到的结果可以改写为$$m_N = \beta S_N \Phi^T t \ \ ; \ \ S_N^{-1} = \alpha^{-1}I + \beta \Phi^T \Phi$$, 我们可以看到当$$\alpha$$趋近于0时，$$m_N \rightarrow w_{ML} = (\Phi^T \Phi)^{-1} \Phi^T t$$。 最大化对数后验概率，得到：

$$\ln p(w \mid t) = -\frac{\beta}{2} \sum_{n=1}^N(t_n - w^T \phi(x_n))^2 - \frac{\alpha}{2}w^Tw +const$$

该式子类似于加了正则项的最大似然估计，其中$$\lambda = \alpha / \beta$$

####3.2预测分布
现在我们想要对一个给定的新x，预测t的值(下式中用c表示新预测值，t表示训练数据)，则可以:

$$p(c \mid t,\alpha,\beta) = \int p(c \mid w, \beta) p(w \mid t,\alpha, \beta)dw$$

其中，条件分布$$p(c \mid w,\beta) = N(t \mid y(x,w),\beta^{-1})$$,后验分布$$p(w \mid t,\alpha,\beta) = N(w \mid m_N, S_N)$$。最终预测分布也是高斯分布，即$$p(c \mid x,t,\alpha,beta) = N(c \mid m_N^T \Phi(x), \sigma_N^2(x))$$， 其中$$\sigma_N^2$$如下：

$$\sigma_N^2(x) = \underbrace{\beta^{-1}}_{noise\ in \ data} + \underbrace{\Phi(x)^TS_N\phi(x)}_{uncertainty \ in \ w}$$

同样可以看到当N趋近于无穷大时，后一项趋近于0。同样，在如果我们使用局部基函数，如高斯，那么在远离基函数中心的区域，第二项中的预测方差的贡献将变为零，只留下$$\beta^{-1}$$作为噪声贡献。因此，预测变得非常有信心，这通常不是一个好的行为，可采用一种替代贝叶斯方法来避免这个影响，用高斯过程方法（在第六章会有介绍）。同样，如果$$w,\beta$$均未知，那么先验分布可以选择Gaussian-gamma分布，最终预测分布是学生分布，这个可以在第二章看到。

####3.3等价核
通过上面的分析，预测分布的均值为$$y(x,m_N) = m_N^T \phi(x) = \beta \phi(x)^T S_N \Phi(x)t = \sum_{n=1}^N \beta \phi(x)^T S_N \phi(x_n)t_n$$,这里的$$S_N^{-1} = S_0^{-1} + \beta \Phi^T \Phi$$. 这里看到了待预测点x与训练数据集中$$x_n$$的内积形式，我们用一个函数来表示，即$$y(x,m_N) = \sum^N_{n=1} k(x,x_n)tn$$,该函数是$$k(x,x') = \beta \phi(x)^T S_N \phi(x')$$，也称之为Smoother matrix, equivalent kernel（这里简单的翻译为等价核）。在解析解中，我们简单的（没有引入正则项）看，$$w = (X^TX)^{-1}X^Tt$$，那么预测x值就是$$wx = \frac{X^Txt}{X^TX}$$，分子是点积形式，分母可以认为是归一化，这里就可以和上面提到的$$m_N,S_N$$对应起来。在线性回归里，可以看到预测值是训练数据集中目标值得线性组合的产生的，也称之为linear smoother。下图是选用多项式基函数和sigmoid基函数的核函数曲线图。

<img src="http://chrispher.github.com/images/prml/ch3_kernel_regression.jpg" height="100%" width="100%">

从直观上来看，核方法是类似于相似度度量，距离预测点y(x')越近的训练目标值t的的权重越大。我们计算一下y(x)和y(x')的协方差，以方便我们更深入的理解等价核。可以看到：

$$\begin{align} cov[y(x),y(x')] &= cov[\phi(x)^w, w^T\phi(x')] \\ &= \phi(x)^T cov(w,w) \phi(x')  \\ &= \phi(x)^T S_N \phi(x') \\ &= \beta^{-1}k(x,x') \end{align}$$

这里使用了w是方差为$$S_N$$的高斯分布。由此看到，距离点预测均值点越近的点相关性越大。此外，通过预测分布生成的点的不确定性是由预测分布的方差决定的（下图3.8）；但是通过w的后验概率分布生成的点，绘制y(x,w)的曲线图，在x点出的y值得不确定性却是由等价核决定的（下图3.9）。

<img src="http://chrispher.github.com/images/prml/ch3_LinearRegression_figure3.8.jpg" height="100%" width="100%">
<img src="http://chrispher.github.com/images/prml/ch3_LinearRegression_figure3.9.jpg" height="100%" width="100%">

<a name="4.贝叶斯模型比较"/>

###4.贝叶斯模型比较

过拟合问题可以通过边缘化模型参数来避免，在使用贝叶斯方法中，交叉验证不再有用，我们可以使用全部数据来训练模型，根据所以训练数据来比较模型。贝叶斯观点下的模型比较，是在模型选择中，使用概率来表示不确定，使用一些概率的加法和乘法原则。假设我们要比较L个模型$${M_i}$$，我们假设数据是由这里的某个模型产生的，但是不确定是具体哪一个，这种不确定性用先验概率$$p(M_i)$$表示。对于给到的数据集D，我们希望评估$$p(M_i \mid D) \propto p(M_i)p(D \mid M_i)$$。我们假设所有模型的先验分布是一样的，那么最终需要关注的就是模型的evidence项$$p(D \mid M_i)$$（有时也称之为边缘似然，marginal likelihood）。此外，$$p(D \mid M_i) / p(D \mid M_i)$$被称之为贝叶斯因子(Bayes Factor)。

我们知道了预测分布，结合概率加法乘法原则,得到右侧的混合分布，如下：

$$p(t \mid x,D) = \sum^L_{i=1} p(t \mid x, M_i, D)p(M_i \mid D)$$

这是一个混合分布(mixture distribution)，是加权平均了各个预测分布$$p(t \mid x, M_i, D)$$的预测值，权重是后验分布$$p(M_i \mid D)$$。比如如果只有两个模型，第一个预测是单峰分布，峰值在t=a,第二个预测也是单峰分布，峰值t=b,根据上式得到的是一个双峰分布（bimodal distribution），而不是一个峰值为(a+b)/2单峰分布。
模型选择就是从这么多模型中选择一个最合适的模型，而不是用所有模型的混合结果（从上式可以看出，如果知道各个模型的先验分布，那就不用做模型选择，直接把各个模型的结果根据上式求和出最终的预测值，类似于Ensemble的方法了）。对于一个受控于一系列参数w的模型，模型的evidence（边缘似然）就是$$p(D \mid M_i) = \int p(D \mid w,M_i)p(w \mid M_j)dw$$。

我们深入的理解一下边缘似然。假设只有一个参数w，它的先验分布是在$$w_{MAP}$$快速达到峰值,宽度为$$\Delta w_{posterior}$$的分布（见下图左侧图）。我们继续假设先验分布是宽度为$$\Delta w_{prior}$$的均匀分布，那么$$p(w) = 1 / \Delta w_{prior}$$，那么我们有（为方便显示，忽略$$M_i$$）：

$$p(D) = \int p(D \mid w) p(w)dw \approx p(D \mid w_{MAP}) \frac{\Delta w_{posterior}}{\Delta w_{prior}}$$

取对数可以得到：

$$\ln p(D) \approx \ln p(D \mid w_{MAP}) + \ln(\frac{\Delta w_{posterior}}{\Delta w_{prior}})$$

如下图左侧图所示，这种近似后的结果。第一项是表示用最优参数下拟合数据的概率分布；第二项表示对模型复杂的惩罚。因为$$\Delta w_{posterior} \lt \Delta w_{prior}$$，即这一项就是负的，$$\Delta w_{posterior} / \Delta w_{prior}$$变得越小，该项值增幅越大。即如果在后验分布中，参数很好的吻合了数据，那么这一项的惩罚越大。

<img src="http://chrispher.github.com/images/prml/ch3_bayes_model_select.jpg" height="100%" width="100%">

如果是有M个参数，同上面那些假设，此外还假设所有参数均匀相同的$$\Delta w_{posterior} / \Delta w_{prior}$$，那么有：

$$\ln p(D) \approx \ln p(D \mid w_{MAP}) + M\ln(\frac{\Delta w_{posterior}}{\Delta w_{prior}})$$

可以看到，M控制着模型的复杂度，如果M很大，那么第一项就会减小，因为复杂的模型能够更好的拟合数据，而第二项将会增大。我们可以根据最大evidence来得到最优的模型复杂度，在之后会用高斯分布来具体推导。此外，我们根据上图右侧图（注意概率和为1，即各个面积相同），来进一步理解模型复杂度和贝叶斯模型选择。这里$$M_1,M_2,M_3$$依次增大，想象一下用这些模型生成数据，越复杂的模型，越能生成不同的数据。具体生成过程是，从参数先验分布p(w)生成一个w，之后根据$$p(D \mid w)$$生成数据。

我们看到了贝叶斯方法下能够避免模型过拟合也能做模型选择。但是，它也是建立在大量的假设之上。实际过程中，因为各种假设很难满足或者很难获取，导致一些错误的结论。因此实际中，预留一部分数据做测试数据是比较明智的。

<a name="5.Evidence近似"/>

###5.Evidence近似

在使用贝叶斯方法的时候，有一些超参数，比如$$\alpha, \beta$$，那么预测分布如下：

$$p(c \mid t) = \int \int \int p(c \mid w,\beta) p(w \mid t,\alpha,\beta)p(\alpha,\beta \mid t)dw d \alpha d \beta$$

那我们如何确定这些参数呢？这里使用的是最大参数w的边际似然（marginal likelihood function），即最大化$$p(\alpha,\beta \mid t)  \propto p(t \mid \alpha,\beta) p(\alpha,\beta)$$得到$$(\hat{\alpha}, \hat{\beta})$$，这种方法也称之为“经验贝叶斯”(empirical Bayes)。我们知道：

$$\ln p(t \mid \alpha,\beta) = \frac{M}{2} \ln \alpha + \frac{M}{2} \ln \beta - E(m_M) - \frac{1}{2} \ln \mid S_N^{-1} \mid - frac{N}{2} \ln(2 \pi)$$

假设$$p(\alpha,\beta \mid t)$$的峰值在$$(\hat{\alpha}, \hat{\beta})$$，那么有：

$$p(c \mid t) \approx p(c \mid t,\hat{\alpha}, \hat{\beta}) = \int p(c \mid w,\hat{\beta}) p(\hat{\alpha}, \hat{\beta}) dw $$ 

我们绘制evidence $$\ln p(t \mid \alpha, \beta)$$对模型复杂度M的图(如下图所示)：
<img src="http://chrispher.github.com/images/prml/ch3_evidenceVSm.jpg" height="100%" width="100%">

我们看到在M=3的时候，值最大，模型最好。通过最大化$$p(t \mid \alpha,\beta)$$，我们对$$\alpha$$求偏导，可以得到 $$\gamma = \sum_i \frac{\gamma_i}{\alpha + gamma_i}$$，最终得到$$\alpha = \frac{\gamma}{m_N^Tm_N}$$。同样，可以得到$$\frac{1}{\beta} = \frac{1}{N-\gamma} \sum^N_{n=1}(t_n - m_N^T \Phi(x_n))^2$$。注意这里得到的$$\alpha,\beta$$都不是最终值，因为$$m_N$$的计算依赖于他们，所以这些是迭代公式。最终我们也能够得到$$\gamma$$的表达式$$\gamma = \alpha m_N^Tm_N$$。$$\gamma$$有一个非常好的解释，即可以解释为有效参数。

最后说一下基函数的一些局限性。本章使用了一些非线性基，但是用的还是线性组合，这种参数线性组合的假设，使得结果是有闭合解，也可以方便的使用最小二乘法和贝叶斯方法。选择一个合适的基，可以让我们把输入变量进行非线性变化。但是基函数$$\Phi_j(x)$$是在数据观测之前就选定的，容易导致维灾难(基函数进行幂次数变化)。在后续章节会陆续看到如何解决这些问题。