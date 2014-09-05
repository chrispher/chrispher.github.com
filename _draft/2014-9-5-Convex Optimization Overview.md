---
layout: post
title:Convex Optimization Overview
comments: true
---
凸优化的概述——这份笔记主要参考了Stanford的Andrew Ng讲授的公开课 《机器学习》的课件。原作者Zico Kolter，由Honglak Lee更新。

### 目录
<!-- MarkdownTOC depth=4 -->
- [1.Introduction](#1.Introduction)
<!-- /MarkdownTOC -->

<a name="1.Introduction" />

### 1.Introduction
在机器学习中，我们需要优化（**optimize**）很多函数，即对于函数$$ f:R^n --> R$$，我们希望找到$$x \in R^n $$ 使得$$f(x)$$最小。在最小二乘法、logistic回归、SVM等方法中都用到了优化的方法。  
通常来说，寻找全局最优是非常困难的。但是，针对于凸优化(convex optimization problems)，是有很多有效的方法。详细可以参考Stanford的教授Stephen Boyd
写的出Convex Optimization以及公开课EE364。
