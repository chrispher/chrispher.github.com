---
layout: post
title: PRML-ch1-Introduction(1)
comments: true
category: PRML
tags: [Machine Learning, Notes, Basic]
---

One of the main goals of this chapter is to introduce, in a relatively informal way, several of the most important of these concepts and to illustrate them using simple examples. This
chapter also provides a self-contained introduction to three important tools that will
be used throughout the book, namely probability theory, decision theory, and information theory.

<!-- MarkdownTOC depth=4 -->
- [1.0 Introduction](#1.0 Introduction)
- [Appendix: No free lunch theorem](#Appendix: No free lunch theorem)

<!-- /MarkdownTOC -->

<a name = "1.0 Introduction"/>

###1.0 Introduction
**Definition**
- **Pattern Recognition**: automatic discovery of regularities in data and the use of these regularities to take actions—classifying the data into different categories.

- **Machine learning**: a sub field of artificial Intelligence. Machine learning focuses on prediction and large scale applications, based on known properties learned from the training data.

- **Data mining**: focuses on the discovery of  (previously) unknown properties in the data. This is the analysis step of Knowledge Discovery in Databases

-**Statistical learning**:a sub field of statistics. Statistical learning emphasizes models and their interpretability, and precision and uncertainty.

**Terminology**:
- trainning or learning phase: determine $$y(x)$$ on the basis of the training data
- test set, generalization
- feature extraction: the original input variables are typically preprocessed to transform them into some new space of variables(for speed or accuracy). Note that new test data must be pre-processed using the same steps as the training data.
- supervised learning: input/taget vectors in the training data
- classification(discrete categories) or regression(continuous variables)
- unsupervised learning: no target vectors in the training data, alse called clustering or **density estimation**. 
- rainforcement learning, credit assignment, exploration, exploitation.

<a name="1.1 Polynomial Curve Fitting"/>

###1.1 Polynomial Curve Fitting
Suppose we observe a real-valued input variable x and we wish to use this observation to predict the value of a real-valued target variable t. Notations are as followed. 

- Training set: $$x=(x_1,...,x_N), t=(t_1,...,t_N)$$
- Goal: predict the taget $$\hat(t)$$ for some new input $$\hat(x)$$
- using linear models: $$y(x,w) = w_0 + w_1x + w_2*x^2 + ... + w_Mx^M = \sum_{j=0}^M w_jx^j$$

**Probability theory provides a framework for expressing such uncertainty in a precise and quantitative manner, and decision theory allows us to exploit this probabilistic representation in order to make predictions that are optimal according to appropriate criteria.** A reasonable choice is to **minimize an error function** that measures the misfit between the function $$y(x,w)$$ and the targets. One simple choice of error function is given by the sum of the squares of the errors. So we minimize $$ E(w) = \frac{1}{2} \sum_{n=1}^N \{y(x_n, w) - t_n)\}^2 $$.(factor $$\frac{1}{2}$$ is included for later convenience).

Choosing the order M called  *model comparision* or *model selection*. However, the fitted curve oscillates wildly(fit the trainning data perfectly.) and gives a very poor representation of the target in testing data -- This is known as *over-fitting*.

**the least squares approach to finding the model parameters represents a specific case of maximum likelihood**, and the over-fitting problem can be understood as a general property of maximum likelihood. By adopting a Bayesian approach, over-fitting problem can be avoided. And there is no difficulty from a Bayesian perspective in employing models for which the number of parameters greatly exceeds the number of data points. Indeed, in a Bayesian model the effective number of parameters adapts automatically to the size of the data set.

<a name = "1.2 Probability theory"/>

###1.2 Probability theory
discrete random variables:
- Sum rule: $$ p(X) = \sum_Y P(X,Y) $$
- Product rule: $$ p(X, Y) = p(X|Y)p(Y) = p(Y|X)p(X) $$
- Bayes: $$ p(Y|X) = \frac{p(X|Y)p(Y)}{\sum_Y p(X|Y)p(Y)} $$, posterior = likelihood \* prior / normalization

continue random variables
- Probability that x lies in an interval:$$ p(x \in (a,b)) = \int_{a}_{b} p(x)d(x)$$
- $$p(x)$$ is called the probability density over x
- $$p(x) leq 1, p(x \in h)


###Appendix: No free lunch theorem
**All models are wrong, but some models are usefull. --George Box**

Much of machine learning is concerned with devising different models, and different algorithms to fit them. We can use methods such as cross validation to empirically choose the best method for our particular problem. However, there is no universally best model — this is sometimes called the no free lunch theorem (Wolpert 1996). The reason for this is that a set of assumptions that works well in one domain may work poorly in another.

As a consequence of the no free lunch theorem, we need to develop many different types of models, to cover the wide variety of data that occurs in the real world. And for each model, there may be many different algorithms we can use to train the model, which make different speed-accuracy-complexity tradeoffs. It is this combination of data, models and algorithms.