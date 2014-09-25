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
- [1.1 Machine Learning](#1.1 Machine Learning)

<!-- /MarkdownTOC -->


#### No free lunch theorem
**All models are wrong, but some models are usefull. --George Box**

Much of machine learning is concerned with devising different models, and different algorithms
to fit them. We can use methods such as cross validation to empirically choose the best method
for our particular problem. However, there is no universally best model — this is sometimes
called the no free lunch theorem (Wolpert 1996). The reason for this is that a set of assumptions
that works well in one domain may work poorly in another.

As a consequence of the no free lunch theorem, we need to develop many different types of
models, to cover the wide variety of data that occurs in the real world. And for each model,
there may be many different algorithms we can use to train the model, which make different
speed-accuracy-complexity tradeoffs. It is this combination of data, models and algorithms.