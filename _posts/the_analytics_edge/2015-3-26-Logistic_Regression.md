---
layout: post
title: TAE：logistic回归
category: 统计分析
tags: [TAE, 数据分析]
description: The Analytics Edge课程下的线性分类分析入门。
---

The Analytics Edge系列的四节笔记，这是第三节：线性回归分析入门，本节主要简述了logistic回归，同时涉及了混淆矩阵、ROC曲线以及决策边界调整(threshold)的内容。

<!-- more -->

系列笔记主要根据the analytics Edge by MITx  课件做的笔记。这门课程，主要通过各个案例来学习数据分析方法。

###目录
{:.no_toc}

* 目录
{:toc}


###课堂简要
本节课主要简述了logistic回归，同时涉及了混淆矩阵、ROC曲线以及决策边界调整(threshold)的内容。在分类过程中，查看混淆矩阵，相比于直接看准确率而言，内容要丰富的多。这里，在R语言使用中，需要一定的技巧，有一定的难度。这里，主要分享一个案例，以及在R语言使用的一些记录。

###案例一 预测贷款偿还
- ####背景
该案例主要分析了贷款人是否会偿还贷款。[数据集](https://courses.edx.org/c4x/MITx/15.071x/asset/loans.csv)来自[LengdingClub.com](https://www.lendingclub.com/info/download-data.action),在数据集中,一共包含了9578个样例、14个变量。主要变量说明如下：  
**credit.policy**: 判断用户是否满足LendingClub.com的信贷承销标准;   
**purpose**: 用户借贷目的，如信用卡、教育、债务整合等;  
**int.rate**: 贷款利率;  
**fico**: FICO指数;  
此外，其他的还包括债务比、过去的贷款记录等等。

- ####认识数据
首先，我们看一下预测变量（是否全部还款）发现，16%的人没有全部还款（可能部分还款，但在这里均认为未还款）。  
此外，我们通过`str(), summary()` 命令看到，部分属性存在缺失值。在这里是一个非常重要的一点，就是缺失值是否需要删除。因为我们要预测所有的借款人，而很多借款人的信息可能不是完整的，因此，我们不会全部删除缺失值。而是根据**自变量**的值，进行预测填充，而不涉及因变量。这里，使用了包`mice`,命令如下：  
`library(mice)`  
`set.seed(144)`  
`vars.for.imputation = setdiff(names(loans), "not.fully.paid")`  
`imputed = complete(mice(loans[vars.for.imputation]))`  
`loans[vars.for.imputation] = imputed`  
之后，我们分割数据为训练集和测试集合。这里使用了包`caTool`, 命令如下:  
`library(caTools)`  
`set.seed(144)`  
`split = sample.split(loan$not.fully.paid, SplitRatio = 0.7)`  
`loanTrain = subset(loan, split == TRUE)`  
`loanTest = subset(loan, split == FALSE)`
- ####建模与分析
- 采用`glm()`命令进行建模，其中`family=binomial`。之后，是对参数系数和P值得一些列认知和解释。这里需要特别注意的是，直接用系数乘以输入，得到的并不是分类结果，而是一个比率A，实际分类概率是`$\frac{1}{(1+e^A)}$`。  
- 之后在预测过程中，注意predict的参数`type=response`可以得到分类的概率值，使用table，得到混淆矩阵。说实话，这些代码挺不容易记住的，这里就贴上来，方便以后直接复制黏贴。如下：  
`pred1 = predict(mod1, type="response")`  
`table(loanTest$not.fully.paid, pred1 >= 0.5)`  
我们可以根据混淆矩阵计算精度、敏感度等等指标（不同的地方，叫法不一致）。同时，计算AUC等等。计算AUC如下(数据集为其他数据集)：  
`library(ROCR)`  
`ROCRpred = prediction(predictTrain, qualityTrain$PoorCare)`  
`ROCRperf = performance(ROCRpred, "tpr", "fpr")`  
`as.numeric(performance(ROCRpred, "auc")@y.values)`  
- 这里提到一个非常重要的核心的问题：**样本有偏**。在认识数据中，我们发现不还款的比例有16%(在其他的案例，可能只有10%左右)，这样我们只是猜测的话，也能够达到84%的准确率——但是，这样就毫无意义了。这里，我们可能会因为业务原因而更加关注某些指标，比如这里，我们更在乎是不是所有的不还款的人都被预测出来了，因此我们可以降低threshold（比如为调整为0.3，这样以前概率为0.3~0.5的人，由预测为还款转为了不还款），尽管这么做使得我们认为一些能够还款的人被认为是不还款了，但是这降低了风险。而在其他的一些案例中，我们可能提高threshold，这样使得我们有更大的把握认为这些一定不还款等等。
- 然而，上一段中的说法虽然非常重要，但不是我做笔记的主要目的。这里，最核心的是**我们要做什么？**，仅仅是为了看一个人还款不还款？案例引导我们进一步思考：**盈利**！这里，不再详细给出如何做，而是给出一个思路。首先我们根据模型，可以预测一个是否全部还款的概率。而在计算盈利的公式中，考虑利率与风险的平衡。这里，选择了利率较大的一部分群体（至少是15%），对于高收益而言，意味着高风险。我们要在高收益中，控制风险，使得损失最小。很遗憾，这里没有采用最优化来求解最合理的概率值。理论上，应该采用最优化的。这里，他选择了模型预测结果不还款概率最低的100人。这样求解了一个综合的收益。

###其他案例与感想
在其他案例中，涉及了一些根据P值进行特征选择，以及上面提到的如何灵活的调整threshold来规避风险。当然，这里特别需要注意的是如何根据业务需求来调整threshold。比如在用户流失中，我们可能要尽可能的捕获流失用户，但是考虑到维护成本，我们希望模型尽可能捕获一定会流失的用户，使得维护成本尽可能的低。
