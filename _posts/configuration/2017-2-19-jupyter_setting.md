---
layout: post
title: 我的juyter notebook设置
category: 环境配置
tags: [软件配置]
description: juyter notebook设置
---

平时经常用jupyter notebook做一些调研和测试代码的工作, 每次换机器都需要重新部署一下环境(主要是复制一下配置), 所以记录一下.

<!-- more -->

### 目录
{:.no_toc}

* 目录
{:toc}

### 1. 生成默认配置
在shell中输入 `jupyter notebook --generate-config` 之后, 会默认在`.jupyter/`下创建一个 jupyter_notebook_config.py 的文件,包含了默认的一些参数.
常见的修改几个配置
``` python
c.IPKernelApp.pylab = 'inline'
c.NotebookApp.ip = '127.0.1.1'
c.NotebookApp.port = 9999
c.NotebookApp.open_browser = False
c.NotebookApp.password = u'sha1:37c8ec50af8b:192a1eb5ca9e1045fbe9e3b60fe52e0844dd1399'

# 以下后缀的文件都会被隐藏
c.ContentsManager.hide_globs=['__pycache__', '*.pyc', '*.pyo', '.DS_Store',
                              '*.so', '*.dylib', '*~', '*.log', '*.pdf']
```
其中, 创建登录密码的方式如下:

``` python
In [1]: from IPython.lib import passwd
In [2]: passwd() # 比如密码设置为1234
Enter password:
Verify password:
Out[2]: 'sha1:37c8ec50af8b:192a1eb5ca9e1045fbe9e3b60fe52e0844dd1399'
```
运行如下命令, 可以启动服务了:
`jupyter notebook --config=/home/datakit/.jupyter/jupyter_notebook_config.py`


### 2.更改字体和样式
##### 2.1 nbconfig
在`.jupyter/`下新建一个文件夹`nbconfig`, 在该文佳夹下,新建一个文件edit.json, 主要是修改jupyter的编辑部分样式
```
{
    "Editor": {
        "codemirror_options": {
            "indentUnit": 4, // 缩进
            "vimMode": false,
            "keyMap": "sublime"
        }
    }
}
```
上述代码是默认4个字符的缩进,使用sublime的快捷键. 详细的一些配置,可以参考官网进一步添加.

#### 2.2 字体
在`.jupyter/`下新建一个文件夹`custom`, 在该文件夹下新建一个文件custom.css, 这个主要是设置jupyter的css样式, 主要是字体和生成样式. 在custom.css添加如下代码:

```
.CodeMirror pre {
    font-family: "monaco";
    font-size: 10pt;
    line-height: 1.5;
}
.container { width:100% !important; }
```

上述主要是设置一下代码的字体和大小, 其他的css样式都可以添加一下.

### 3.jypyter的插件

####  3.1 隐藏code区域

``` html
from IPython.display import HTML
HTML('''<script>
code_show=true;
function code_toggle() {
if (code_show){
$('div.input').hide();
} else {
$('div.input').show();
}
code_show = !code_show
}
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
```

#### 3.2 在markdown中使用变量

其实在ipython notebook中有大量的扩展库可以使用，可以参考[Github](https://github.com/ipython-contrib/IPython-notebook-extensions)。这里举一个例子，就是在markdown的cell使用{{a}} 来表示变量a的值，使得在写报告的时候，可以直接与变量交互。

复制扩展库里的usability里python-markdown里的python-markdown.js(github里面是main.js) 到 nbextensions 目录里(默认的在用户的文件下的ipython里)。这里也给放到其他路径，但是需要指定路径。之后在ipython的cell里输入

``` sh
%%javascript
IPython.load_extensions('python-markdown')

# 之前code的cell里，设定 a = 25
# markdown 里的 input: 今年成交额是{{a}}个亿
# markdown 里的 output: 今年成交额是{{25}}个亿
```
