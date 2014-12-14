---
layout: post
title: matplotlib之个人常用绘图
category: Python
tags: [python, 绘图, matplotlib]
---

本文主要关于Python下的绘图包matplotlib的使用，涉及几个自己在实际过程中经常用的一些绘图命令。在这里，对个人经常使用的绘图功能做一个简单的总结，以便以后查看和使用！因为是在建模过程中的作图，所以不是追求论文中作图的那种详尽，而是尽可能用简短的命令实现所需要的绘图。本文所有代码默认`from matplotlib import pyplot as plt` 和 `import numpy as np` 。此外，用到一些不是很常用的绘图可以去 [matplotlib gallery](http://matplotlib.org/gallery.html)和[matplotlib doc](http://matplotlib.org/contents.html#)查看。

<!-- more -->

### 目录
- [1.使用风格](#[1.使用风格)
- [2.subplots](#[2.subplots)
- [3.colors](#3.colors)
- [4.matrix](#4.matrix)

<a name="1.使用风格"/>

###1.使用风格

很多人说matplotlib绘图和matlab一样，配色很难看。其实，是他们不会用。matplotlib提供多种配色，可以`print plt.style.available
`, 看到有 `[u'dark_background', u'bmh', u'grayscale', u'ggplot', u'fivethirtyeight']` 五种独特的配色方案。代码和结果如下：

{% highlight Python %}

x = np.random.random((100,1))
plt.style.use('ggplot')
plt.plot(np.sin(np.linspace(0, 2*np.pi)))
plt.title('ggplot style')
plt.show()

with plt.style.context(('bmh')):
    plt.plot(np.sin(np.linspace(0, 2*np.pi)))
    plt.title('bmh style')
plt.show()

{% endhighlight %}

<img src="http://chrispher.github.com/images/python/matplotlib_style.jpg" height="70%" width="70%">

<a name="2.subplots"/>

###2.subplots
subplots使用还是比较多的。如果子图比较多，可以使用for循环；如果比较少，可以直接使用；代码和结果如下：

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# ax = plt.subplot(2, 2, 1) for the particular one
fig, axes = plt.subplots(ncols=2, nrows=2)
ax1, ax2, ax3, ax4 = axes.ravel()

# scatter plot (Note: `plt.scatter` doesn't use default colors)
x, y = np.random.normal(size=(2, 200))
ax1.plot(x, y, 'o')

# sinusoidal lines with colors from default color cycle
L = 2*np.pi
x = np.linspace(0, L)
ncolors = len(plt.rcParams['axes.color_cycle'])
shift = np.linspace(0, L, ncolors, endpoint=False)
for s in shift:
    ax2.plot(x, np.sin(x + s), '-')
ax2.margins(0)

# bar graphs
x = np.arange(5)
y1, y2 = np.random.randint(1, 25, size=(2, 5))
width = 0.25
ax3.bar(x, y1, width)
ax3.bar(x+width, y2, width, color=plt.rcParams['axes.color_cycle'][2])
ax3.set_xticks(x+width)
ax3.set_xticklabels(['a', 'b', 'c', 'd', 'e'])

# circles with colors from default color cycle
for i, color in enumerate(plt.rcParams['axes.color_cycle']):
    xy = np.random.normal(size=2)
    ax4.add_patch(plt.Circle(xy, radius=0.3, color=color))
ax4.axis('equal')
ax4.margins(0)

plt.show()

{% endhighlight %}

<img src="http://chrispher.github.com/images/python/plot_ggplot.png" height="70%" width="70%">

<a name="3.colors"/>

###3.colors

```
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
```

<a name="4.matrix"/>

###4.matrix

矩阵绘图是比较多的，可以画混淆矩阵，也可以画相关系数矩阵等！代码和结果如下：
{% highlight python%}
plt.style.use('ggplot')
m = np.random.rand((10,10))
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

{% endhighlight%}

<img src="http://chrispher.github.com/images/python/plot_matrix.png" height="70%" width="70%">
