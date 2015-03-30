---
layout: post
title: ch4-线性分类之生成模型
category: 机器学习
tags: [机器学习, PRML]
description: 概述一下线性分类相关的一些知识, 本文主要是生成模型，包括Losgistic回归等
---

本篇主要是概述一下线性分类相关的一些知识。分类的目标是在给到一个D维的输入数据X，预测它的类别c（假设共K个类别，一般而言，各个类别是互斥的）。输入空间可以被分割为不同的决策区域(decision regions),这些区域的平面称之为决策边界(decision boundaries或decision surfaces，本文使用决策平面)。这一章，主要考虑线性分类器，即决策平面是输入x的线性模型。如果数据可以完全被线性决策平面分割，称之为线性可分（linearly separable）。笔记分三部分，这是第二部分：生成模型为主部分。

<!-- more -->

注：本文仅属于个人学习记录而已！参考Chris Bishop所著[Pattern Recognition and Machine Learning(PRML)](http://research.microsoft.com/en-us/um/people/cmbishop/PRML/)以及由Elise Arnaud, Radu Horaud, Herve Jegou, Jakob Verbeek等人所组织的Reading Group。

###目录
{:.no_toc}

* 目录
{:toc}


###1.生成模型
上一节简单的说明了不同类型的分类器。这里我们使用生成模型，需要得到类别的先验分布$$p(C_k)$$和条件概率密度$$p(x \mid C_k)$$，之后使用贝叶斯法则得到后验概率分布$$p(C_k \mid x)$$。考虑一下二元分类，我们有：

$$p(C_1 \mid x) = \frac{p(x \mid C_1)p(C_1)}{p(x \mid C_1)p(C_1) + p(x \mid C_2)p(C_2)} = \frac{1}{1 + exp(-a)} = \sigma(a)$$

这里我们定义$$a = \ln \frac{p(x \mid C_1)p(C_1)}{p(x \mid C_2)p(C_2)}$$，而$$\sigma(a)$$就是我们熟知的sigmoid函数。它的逆函数为$$a = \ln \frac{\sigma}{1 - \sigma}$$，表示概率比值的对数。那么对于K大于2的分类而言，如下：

$$p(C_k \mid x) = \frac{p(x \mid C_k)p(C_k)}{\sum_j p(x \mid C_j)p(C_j)} = \frac{exp(a_k)}{\sum_j exp(a_j)}$$

这里我们定义$$a_k=\ln p(x \mid C_k)p(C_k)$$。这个就是归一化指数(normalized exponential),一般都是称softmax函数，因为它表示了一个平滑的最大函数，即如果对于所有$$j \ne k$$，有$$a_k >> a_j$$，那么$$p(C_k \mid x) \approx 1; \ p(C_j \mid x) \approx 0$$。

接下来就是如何得到这些条件概率值，分别从连续值输入和离散值输入两方面说。

####1.1连续输入

对于连续输入，我们可以假设不同类别下条件概率均服从高斯分布，且共享一个协方差矩阵。那么，我们有：

$$p(x \mid C_k) = \frac{1}{(2 \pi)^{D/2}} \frac{1}{\mid \Sigma \mid ^{1/2}} exp(-\frac{1}{2}(x -\mu_k)^T \Sigma^{-1} (x - \mu_k))$$

结合上面的后验概率和sigmoid公式，二元分类下有$$p(C_1) = \sigma(w^Tx + w_0)$$，这里有$$w = \Sigma^{-1}(\mu_1 - \mu_2); \  w_0 = -\frac{1}{2}\mu_1^T \Sigma^{-1} \mu_1 + \frac{1}{2}\mu_2^T \Sigma^{-1} \mu_2 + \ln \frac{p(C_1)}{p(C_2)}$$。

我们再看一下决策边界，是$$p(C_k \mid x)$$均一致的地方，这个受x的线性函数控制，因此决策边界是线性的。此外，先验概率$$p(C_k)$$的影响是$$w_0$$来平行移动决策边界的。那么对于多元分类，同理可得$$a_k(x) = w^T_k x + w_0$$, 这里有$$w_k = \Sigma^{-1} \mu_k; \ \ w_{k0} = - \frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k + \ln p(C_k)$$。同样，这也是一个广义的线性模型。

如果我们放松约束，使得各个类别的条件概率$$p(x \mid C_k)$$有各自的协方差矩阵$$\Sigma_k$$，那么之前的结论就变了，无法像之前那样消掉二次项了，决策边界也会呈现二次型。

一旦我们选用了特定分布，那么有需要进行参数估计，这里拿二元分类说一下，采用最大似然的方法。这里数据集是$$(x_n,t_n)$$，先验概率$$p(C_1) = \pi$$，推导比较直接，把相关分布代入，可以得到：

$$p(t \mid \pi,\mu_1,\mu_2,\Sigma) = \prod^N_{n=1}(\pi N(x_n \mid \mu_1,\Sigma)^{t_n}((1-\pi) N(x_n \mid \mu_2,\Sigma))^{1-t_n}$$

之后我们取对数，对$$\pi$$取导数为0，可以得到$$\pi = \frac{1}{N} \Sigma t_n = \frac{N_1}{N}$$,这里$$N$$是样本总数，$$N_1$$是第一类样本数。对$$\mu_1$$取导数为0，可以得到$$\mu_1=\frac{1}{N_1} \Sigma^N_{n=1} t_n x_n$$,同理$$\mu_2=\frac{1}{N_2} \Sigma^N_{n=1} (1-t_n)x_n$$（即各自类别下的均值）。对共享的$$\Sigma$取导数为0，可以得到：

$$\Sigma = S = \frac{N_1}{N}S_1 + \frac{N_2}{N}S_2 = \frac{1}{N_1} \sum_{n \in C_1}(x_n - \mu_1)(x_n - \mu_1)^T +  \frac{1}{N_2} \sum_{n \in C_2}(x_n - \mu_2)(x_n - \mu_2)^T$$

这里很容易扩展到K个类别。注意：我们使用了高斯分布，对异常值很敏感。

####1.2离散输入
如果是离散的输入，这里以二元变量为例，即$$x_i \in (0,1)$$，如果输入有D维度，为了覆盖到每一个类别，至少要$$2^D -1$$个独立变量（包含一个求和约束），这是一个随着维度而指数增长的！我们需要找一个更受限制的表达。这里引入了朴素贝叶斯假设(Naive Bayes assumption)，即假设各个类别下的特征是相互独立的，那么条件分布形式为$$p(x \mid C_k) = \prod^D_{i=1} \mu^{x_i}_{ki}(1-\mu_{ki})^{1-x_i}$$，只包含D个独立变量。那么softmax里面那个$$a_k$$公式可以细化为:

$$a_k(x) = \sum^D_{i=1} (x_i \ln \mu_{ki} + (1-x_i) \ln (1-\mu_{ki})) + \ln p(C_k)$$

####1.3指数族
之前已经看到了高斯分布和离散输入了，类别的后验概率可以用线性模型和sigmoid函数(K=2)或softmax(K > 2)的结合得到。然而这些只是指数族的一个特例。我们假设类别下的条件概率密度$$p(x \mid C_k)$$是指数族里的一个分布，那么有$$p(x \mid \lambda_k) = h(x)g(\lambda_k)exp(\lambda_k^T u(x))$$。接下来我们限制一下，使得$$u(x) = x$$，之后引入一个缩放参数s，那么限制后的指数族的条件分布是:

$$p(x \mid \lambda_k,s) = \frac{1}{s} h(\frac{1}{s} x) g(\lambda_k)exp(\frac{1}{s} \lambda_k^Tx)$$

这里可以看到我们允许各个类有自己的参数向量$$\lambda_k$$，但是假设共享一个参数s。对于二元分类，我们转化得到$$a(x) = (\lambda_1 - \lambda_2)^T x + \ln (g(\lambda_1)) - \ln g(\lambda_2) + \ln p(C_1) - \ln p(C_2)$$
。同样，对于多元分类，$$a_k(x) = \lambda_k^Tx + \ln g(\lambda_k) + \ln p(C_k)$$，同样是x的线性函数。

###2.概率判别模型
上面我们的模型是使用最大似然得到$$p(x \mid C_k)$$和$$p(C_k)$$的参数值，之后使用贝叶斯法则得到后验分布。接下来用的概率判别模型(Probabilistic Discriminative Models)，则是直接使用最大似然估计得到线性模型的参数值。我们也就讨论一个比较有效的优化算法"迭代再加权最小平方法"(IRLS, iterative reweighted least squares)。与其间接的通过贝叶斯法则得到后验概率，我们直接拟合后验概率分布，这种方法称之为概率判别法，一般称之为判别法，优点是有更少的参数，而且可能会有更好的预测效果，尤其是当假设的条件概率分布$$p(x \ mid C_k)$$与实际差距比较大的时候。

这里同样要说明的一点是，我们可以有不同的基函数，对输入数据x做基函数变化得到模型的输入特征$$\Phi(x)$$，基函数可以是非线性的，但是最终的决策边界是新特征的线性组合，所以还是认为是线性模型。在$$\Phi$$空间下线性可分，在x空间下未必线性可分！同时接下来我们也引入$$\Phi_0(x)=1$$这种表示。

对于许多模型而言，$$p(x \mid C_k)$$都会有很明显的重合，这对应了后验概率不一定是0和1（介于0-1之间的一些值，所以有时候还要选择阈值来决定是0或1的分界）。解决的方法是尽可能准确的得到后验概率，之后使用决策理论。需要注意的是非线性变换$$\Phi(x)$$并不会消除这些类别间的重合。实际上，它甚至会增加重合区域，或者产生新的重合区域。但是，合适的非线性变化能够使得后验概率的计算更加的容易！当然，这些基函数也是有限制的，后面会具体的提到这些不足，并改善。

####2.1Losgistic回归
通过之前的讨论，在一些假设之下有（这里还是讨论二元分类）：$$p(C_1 \mid \Phi) = y(\Phi) = \sigma(w^T \Phi)$$，这里的$$\sigma()$$是logistic函数。在统计学里，这个模型称之为logistic回归。在M维度的特征空间$$\Phi$$里，我们有M个参数。对比一下，如果我们使用最大似然拟合了不同类别下的条件概率密度（高斯分布），那么均值对应2M个参数，共享的协方差矩阵对应M(M+1)/2个参数，结合先验概率一共有M(M+5)/2+1个参数，是随着M增长而二次方增长。现在，我们使用最大似然估计来得到logistic回归的参数。

对于数据集$$(\Phi_n, t_n)$$，这里$$t_n \in (0,1), \Phi_n = \Phi(x_n)$$,似然函数如下：

$$p(t \mid w) = \prod^N_{n=1} y_n^{t_n}(1-y_n)^{1-t_n}$$

这里的$$y_n = p(C_1 \mid \Phi_n)$$，去对数之后得到如下误差函数，也称之为交叉熵(cross-entropy)误差函数：

$$E(w) = -\ln p(t \mid w) = -\sum^N_{n=1}(t_n \ln y_n + (1 - t_n) \ln (1 - y_n))$$

这里$$y_n = \sigma(a_n); \ \ a_n = w^T \Phi_n$$，对w去导数得到：$$\nabla E(w) = \sum^N_{n=1} (y_n - t_n) \Phi_n $$。我们得到了一个非常简单的形式，非常类似于线性回归模型！这里梯度的贡献主要来自“误差”$$y_n - t_n$$(预测值和实际值之间的差距)，乘以基函数向量$$\Phi_n$$。

有了梯度之后，我们可以使用序列学习(在线学习)等方法。当然，当数据线性可分的时候，最大似然可能导致严重的过拟合。这是因为最大似然在超平面取$$\sigma=0.5$$时有解，即$$w^T \Phi =0$$，把两类分开，使得w可以无限增大。这样下来(非常大)，logistic函数就会非常的陡峭，将容易导致训练样本点的后验概率$$p(C_k \mid x)$$为1。此外，结果与优化算法和初始化参数选择有关，但最大似然法无法判断哪个结果更好。这里的过拟合问题跟数据集大小没有关系，只要训练数据线性可分。解决的办法是引入先验概率，采用MAP求解W，或者等价的使用正则项。

####2.2迭代再加权最小平方 
在回归模型中，基于误差服从高斯分布等假设，有了闭合解，这是因为对数似然函数中的二次项依赖于w。在logistic回归中，因为非线性函数sigmoid，而不再有闭合解了，但是误差函数还是凹的，只有唯一的最小值。即可以通过有效的迭代，比如Newton-Raphson(牛顿法)，使用了局部二次项来近似对数似然函数，牛顿法迭代公式如下：

$$w^{(new)} = w^{(old)} - H^{-1} \nabla E(w)$$

这里的H是Hessian矩阵，它的元素是E(w)对w的二阶导数。如果我们在回归中，使用牛顿法，会怎么样呢？如下：

$$\nabla E(w) = \sum^N_{n=1} (w^T \Phi_n - t_n)\Phi_n = \Phi^T \Phi w - \Phi^T t$$

那么我们有：$$H = \nabla \nabla E(w) = \Phi^T \Phi$$，那么有：

$$w^{(new)} = w^{(old)} - (\Phi^T \Phi)^{-1} (\Phi^T \Phi w^{(old)} - \Phi^T t) = (\Phi^T \Phi)^{-1} \Phi^T t$$

这个结果和解析解是一致的。那么我们把这个方法应用到logistic回归模型中，如下：

$$\nabla E(w) = \sum^N_{n=1} (y_n - t_n) \Phi_n = \Phi^T(y - t)$$

$$H = \nabla \nabla E(w) = \sum^N_{n=1} y_n(y_n - t_n) \Phi_n \Phi_n^T = \Phi^T R \Phi $$

这里的R是N维对角矩阵，对角元素$$R_{nn} = y_n (1 - y_n)$$。这里的Hessian矩阵不再固定，而是依赖于w的计算(y的值要通过w计算)。这里Hessian矩阵是正定的，误差函数由全局唯一的最小值。那么，使用牛顿法更新logistic回归权重如下：

$$\begin{align} w^{(new)} &= w^{(old)} - (\Phi^T R \Phi)^{-1} \Phi^T (y - t) \\ &= (\Phi^T R \Phi)^{-1} (\Phi^T R \Phi w^{(old)} - \Phi^T (y - t)) \\ &= (\Phi^T R \Phi)^{-1} \Phi^T R z \end{align}$$

这里z是N维向量，$$z = \Phi w^{(old)} - R^{-1}(y-t)$$。从上面的更新式子看出这是一个加权最小二乘法，权重矩阵是R。每次更新都需要重新根据w得到新的权重矩阵R，所以这个方法也被称之为迭代再加权最小平方(IRLS, iterative reweighted least squares)。 

加权最小二乘法中，对角权重矩阵R在logistic回归中，可以解释为方差$$var(t) = E(t^2) - E(t)^2 = \sigma(x) - \sigma(x)^2 = y(1-y) = R_{NN}$$。此外，IRLS也可以解释为变量$$a = w^T \Phi$$空间里的线性问题的解。而z可以近似的解释为有效目标值（公式见课本）。


####2.3多类别Losgistic回归
拓宽到多类别logistic回归，我们之前已经给出了后验概率分布公式$$p(C_k \mid \Phi) = y_k(\Phi) = \frac{exp(a_k)}{\sum_j exp(a_j)}$$，其中$$a_k = w^T_k \Phi$$。此外，$$\frac{\partial y_k}{ \partial a_j} = y_k(I_{kj} - y_j)$$。这里的I是单位矩阵。接下来就是最大似然求解参数，对数似然函数如：

$$E(w_1,...w_k) = - \ln p(T \mid w_1,...,w_k) = -\sum^N_{n=1} \sum^K_{k=1} t_{nk} \ln y_nk$$

这个多元分类下的误差函数称之为交叉熵( cross-entropy)误差函数。导数为$$\nabla_{w_j} E(w_1,...,w_K) = \sum^N_{n=1} (y_{nj} - t_{nj}) \Phi_n$$。注意这里我们有一个约束$$ \sum_k t_{nk} = 1$$(各个类别下概率和为1，如果没有这个约束，那么w没有固定解，我们也可以用正则项约束来达到唯一解)。 

这里更新的形式和之前二元分类一致的。同样，我们也可以使用牛顿法，对于的Hessian矩阵如下：

$$\nabla_{w_k} \nabla_{w_j} E(w_1,...,w_K) = - \sum^N_{n=1} y_{nk}(I_{kj} - y_{nj}) \Phi_n \Phi_N^T$$

####2.4Probit回归
在上面看到了用指数族分布描述的类别下的条件概率，结果可以在线性特征下的logistic或softmax来得到后验概率。但是，很多情况下类别下的条件概率并不能给出一个简单形式的后验概率。这里我们继续考虑二元分类，仍然是在线性模型的框架下，即$$p(t=1 \mid a) = f(a)$$，这里的$$a = w^T \Phi$$，f是激活函数。那么我们就会有很多选择，这里选择一个在$$a_n \le \theta$$时，$$t_n=1$$，否则$$t_n = 0$$。这里的$$\theta$$是由概率密度$$p(\theta)$$来产生的，那么我们的激活函数可以用累积分布函数（cdf）$$f(a) = \int^a_{- \infty}$$表示。

这里作为一个例子，我们选择标准高斯分布作为$$p(\theta)$$，那么cdf就是$$\Phi(a) = \sum^a_{- \infty} N(\theta \mid 0,1) d \theta$$，这就是probit函数，形状和sigmoid函数类似。这里用其他的高斯分布，不会改变模型，只是相当于对线性系数w做了尺度变换。此外还有一些函数，比如$$erf(a) = \frac{2}{\sqrt{\pi}} \int^a_0 exp(-\theta^2 / 2 d \theta)$$，称之为erf函数或误差函数(与机器学习里的误差函数不同)。与probit函数关系为$$\Phi(a) = \frac{1}{2} (1 + \frac{1}{\sqrt{2}} erf(a))$$。

基于probit激活函数的广义线性模型，称之为probit回归。我们可以使用最大似然估计来得到参数。实际中，probit得到的结果类似于logistic回归的结果。实际应用中需要注意的是异常值。probit回归比logistic回归对异常值更加敏感，因为logistic函数尾部随着x趋近于无穷大，按照$$exp(-x)$$衰减的，而probit是按照类似于$$exp(-x^2)$$的衰减。但是，两个在训练中，都是假设数据是正确标定的。错误标签可以通过引入错误标定的概率$$\epsilon$$，并加入到概率模型中如下：

$$p(t \mid x) = (1 - \epsilon) \sigma(x) + \epsilon (1 - \sigma(x)) = \epsilon + (1 - 2 \epsilon) \sigma(x)$$

这里的$$\sigma(x)$$就是激活函数。而$$\epsilon$$也可以提前设定，也可以作为一个超参数来通过训练得到。


####2.5规范link函数
通过之前的讨论，我们可以看到不管是分类还是回归，线性模型的误差或交叉熵误差对参数的导数都是同一个形式，即'error'项$$y_n - t_n$$乘以特征向量$$\Phi_n$$，这里的$$y_n = w^T \Phi_n$$或$$y_n = f(w^T \Phi_n)$$。我们扩展到：只要目标变量下的条件分布是指数族的分布，且激活函数是规范link函数(Canonical link functions)，那么我们的结果仍是上面形式的。

这里目标变量下的条件分布是指数族分布(上一节1.3指数族里)，即$$p(t \mid \eta, s) = \frac{1}{s} h(\frac{t}{s}) g(\eta) exp(\frac{\eta t}{s})$$，推导一下有:$$y = E(t \mid \eta) = - s \frac{d}{d \eta} \ln g(\eta)$$。这里y是$$\eta$$的函数，反过来就是$$\eta = \phi(y)$$。接下来就是广义线性模型(generalized linear model)，定义为y是一个非线性函数，自变量是输入特征的线性组合，即$$y = f(w^T \Phi)$$。这里$$f$$在机器学习里一般称之为激活函数(activation function),$$f^{-1}$$在统计里称之为link函数(link function)。

那在假设我们共享缩放参数和噪声服从高斯分布情况下，取对数为0，可以得到如下结论:

$$\nabla \ln E(w) = \frac{1}{s} \sum^N_{n=1} (y_n - t_n) \Phi_n$$

对于高斯分布，$$s = \beta^{-1}$$，对于logistic模型$$s = 1$$。
