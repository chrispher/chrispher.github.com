---
layout: post
title: Feature hashing
category: Advanced Machine Learning
tags: [trick, 特征, 哈希]
---

在机器学习中，[Feature hasing](https://en.wikipedia.org/wiki/Feature_hashing) 也称之为[hashing trick](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf)，是一种快速的且很节省空间的特征向量化的方法。

<!-- more -->

###Content
- [1.Motivating example](#1.Motivating example)
- [2.overview](#2.overview)
- [3.Feature vectorization using the hashing trick](3.Feature vectorization using the hashing trick)
- [4.other](#4.other)

<a name="1.Motivating example"/>

###1.Motivating example

在文本分类任务中，我们需要将输入的文本转化成数值向量。一般而言，我们都采用词库（a bag of words, BOW）来构建，之后根据：
比如如下输入：

- John likes to watch movies.
- Mary likes movies too.
- John also likes football.

那么我们得到的词库格式是(词语，索引)是[(John, 1), (likes, 2), (to, 3), (watch, 4), (movies, 5), (Mary, 6), (too, 7),(also, 8), (football, 9)],即有9个特征。对于每一句话，我们只有看看这些特征是否出现在输入语句中即可。那么对于第一句话的vector就是 [1, 1, 1, 1, 1, 0, 0, 0, 0]。有时候也会考虑词语出现次数，比如对于句子*John likes movies and Mary likes movies too*, 如果不考虑次数，对应的vector(注意这里and不在词库里)是[1, 1, 0, 0, 1, 1, 1, 0, 0]。如果考虑次数，则是[1, 2, 0, 0, 2, 1, 1, 0, 0]。

想象一下，如果我们的词库非常大，那么索引起来需要消耗大量的空间和资源。另外，考虑到新的词语(如上面的and)，需要不断的增加词库中的词。而采用哈希(hash)的方法，即对输入进行哈希处理，得到的哈希值直接作为index，而不是像之前那样查找。

注意hashing trick 并不局限在文本分类，也可以用在其他特征较大的问题中。

<a name="2.overview"/>

###2.overview

我们设计一个函数[v = h(x)](http://metaoptimize.com/qa/questions/6943/what-is-the-hashing-trick) , 能够将d维度向量x = (x(1),x(2),...,x(d))转化成m维度的新向量 v，这里的m可以大于也可以小于d。一种方法使用hash(哈希)函数将x(1)映射到v(h(1)), 将x(d)映射到v(h(d))。Hash 函数能够将任意输入转换到一个固定范围的整数输出。好的hash函数函数应该有均匀的输出，并遵守雪崩效应：在输出中的一个小的扰动必须导致在输出上有很大的变化。这确保了在X中维度将被映射到v中的随机维度。注意，这将典型地导致碰撞（collisions, 即x的两个维度可以被映射到v中相同的维度），但在实践中，如果m是足够大，这将不会影响性能。

<a name="3.Feature vectorization using the hashing trick"/>

###3.Feature vectorization using the hashing trick

在文本分类中，hash函数以string(字符串)作为输入，我们需要将这些words映射到所需要的维度v（预先定义的长度）。

{% highlight python %}
    def hashing_vectorizer(s, N):
        x = [0 for i in xrange(N)]
        for f in s.split():
             h = hash(f)
             x[h % N] += 1
        return x

    print hashing_vectorizer('we are the great', 4) # return [0, 1, 3, 0]
    print hashing_vectorizer('tell me the way to the IBM', 4) # return [1, 3, 3, 0]
{% endhighlight %}

如果我们引入一个二值函数g(x)(假设输出为1，-1)来决定更新数值是加还是减，以此来对坑hash collisions(哈希碰撞)。那么算法更新为:

{% highlight python %}
    def hashing_vectorizer(s, N):
        x = [0 for i in xrange(N)]
        for f in s.split():
            h = hash(f)
            x[h % N] += 1*g(f)
        return x
{% endhighlight %}

上面给出了两个比较简单的例子。一个比较优化的方法是生成一组(h,g)并由算法使用，线性模型也可以作为hash表用于特征系数表示。

<a name="4.other"/>

###4.other
另外也有用hash函数自动生成叉乘特征(cross-product features)。比如你的hash函数能够从两个数中生成一个，即i = h(a,b), 那么你就可以把一些组合特征x(a)*x(b)变化为v(i)。Cross-product 在建模一些交互特征中比较有效。

论文 [Feature Hashing for Large Scale Multitask Learning](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf) 也提到用hash方法解决多任务学习问题。包括降维、稀疏表达、叉乘特征、文本特征、多任务学习等等。