---
layout: post
title: caffe深度学习网络结构可视化
category: 深度学习
tags: [环境搭建, caffe]
description: caffe深度学习网络结构可视化
---

之前写得一个小工具，用来可视化caffe里各种prototxt定义的网络。当然，代码比较简单，可以修改一下用户可视化各种相关的网络模型，不依赖于caffe和proto解析。

<!-- more -->

### 目录
{:.no_toc}

* 目录
{:toc}

### 1.code

``` python

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, sys
from graphviz import Digraph


def parse_line(line):
    name = re.findall(r'}name:"(.*?)"', line)
    types = re.findall(r'type:"(.*?)"', line)
    inputs = re.findall(r'bottom:"(.*?)"', line)
    outputs = re.findall(r'top:"(.*?)"', line)
    return name, types, inputs, outputs


def parse_file(filename):
    with open(filename) as f:
        data = f.read()
        data = data.replace("\n", "")
        data = data.replace(" ", "")
        data = data.split("layer{")
    plot_data = []
    for i in xrange(len(data) - 1):
        res = {}
        line = "}" + data[i+1]
        name, types, inputs, outputs = parse_line(line)
        res["name"] = name[0]
        res["types"] = types[0].lower()
        res["inputs"] = inputs
        res["outputs"] = outputs
        plot_data.append(res)
    # print(plot_data)
    return plot_data



def plot(plot_data, width="1.0"):
    g = Digraph("LR", filename='er.gv', engine='dot', format='pdf')
    node_attr = {"shape": "record", "fixedsize": "true",
                "style": "rounded,filled",
                 "color": 'lightblue2',"concentrate":"true"}
    g.node_attr.update(node_attr)
    cm = ("#8dd3c7", "#fb8072", "#bebada", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5")

    size = 1.0
    for l, d in enumerate(plot_data):
        name = d['name']
        outputs = d['outputs']
        e = Digraph(name)

        if "pool" in d['types']:
            e.attr('node', shape='box', color=cm[3])
        elif 'loss' in d['types']:
            e.attr('node', shape='ellipse', color=cm[1])
        elif "convolution" in d['types']:
            e.attr('node', shape='box', color=cm[4])
        elif "concat" in d['types']:
            e.attr('node', shape='box', color=cm[6])
        elif "relu" in d['types']:
            e.attr('node', shape="box", color=cm[2])
        else:
            e.attr('node', shape="box", color=cm[5])

        if len(outputs) < 2:
            name = outputs[0]
        # special for the layer whoes inputs are sample with outputs
        if d['inputs'] == d['outputs']:
            e.node(name, width=width, height="0.6")
        else:
            label= '''<<TABLE BORDER="0">
                      <TR><TD><FONT POINT-SIZE="12">%s</FONT></TD></TR>
                      <TR><TD><FONT POINT-SIZE="8" COLOR="blue">%s</FONT></TD></TR>
                    </TABLE>>''' %(name, d['types'])
            e.node(name, label, width=width, height="0.6")
        e.attr('node', shape='ellipse')
        for i in d['inputs']:
            if 'slot' in i:
                e.node(i, fontsize="8", width="0.5", height="0.4", color=cm[0])
            else:
                e.node(i, fontsize="12", width=width, height="0.6")
            if i == name:
                e.edge(i, name, label=d['types'], fontsize="5", color=cm[0])
            else:
                e.edge(i, name)

        # mult outputs
        if len(outputs) > 1:
            for i in outputs:
                e.node(i, fontsize="12", width=width, height="0.6")
                e.edge(name, i)
        g.subgraph(e)

    g.view()


if __name__ == "__main__":
    filename = sys.argv[1]
    # filename = "googlenet.prototxt"
    res = parse_file(filename)
    width = "1.5"
    plot(res, width)

```


### 2.例子

这里可视化了各种网络结构，比如google的部分如下：

<img src="/images/deeplearning/GoogleNet_part.png" height="80%" width="80%">
