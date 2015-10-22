---
layout: post
title: My sublime text2 setting
category: app_config
tags: [软件配置]
description: windows下sublime text2的一些设置和调整
---

update 2015-10-21：增加一些插件的设置和sublime常用技巧、快捷键等；

windows下sublime text2的一些设置和调整。

<!-- more -->

###目录
{:.no_toc}

* 目录
{:toc}

###1. 安装插件Package Control
在控制台输入以下代码后重启：

{% highlight python %} 
import urllib2,os; pf='Package Control.sublime-package';
ipp = sublime.installed_packages_path();
os.makedirs( ipp ) if not os.path.exists(ipp) else None;
urllib2.install_opener( urllib2.build_opener( urllib2.ProxyHandler( )));
open( os.path.join( ipp, pf), 'wb' ).write( urllib2.urlopen( 'http://sublime.wbond.net/' +pf.replace( ' ','%20' )).read());
print( 'Please restart Sublime Text to finish installation')
{% endhighlight %} 

如果是sublime text3的，如下

{% highlight python %} 
import urllib.request,os; pf = 'Package Control.sublime-package'; 
ipp = sublime.installed_packages_path();
urllib.request.install_opener( urllib.request.build_opener( urllib.request.ProxyHandler()) ); 
open(os.path.join(ipp, pf), 'wb').write(urllib.request.urlopen( 'http://sublime.wbond.net/' + pf.replace(' ','%20')).read())
{% endhighlight %} 

###2. 推荐主题
主题设置,这里使用Afterglow，也可以尝试theme fatland和brogrammer。
{% highlight python %} 
{
    "binary_file_patterns":
    [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.ttf",
        "*.tga",
        "*.dds",
        "*.ico",
        "*.eot",
        "*.swf",
        "*.jar",
        "*.zip"
    ],
    "color_inactive_tabs": false,
    "color_scheme": "Packages/Theme - Afterglow/Afterglow.tmTheme",
    "detect_indentation": true,
    "draw_centered": false,
    "fold_buttons": true,
    "folder_no_icon": false,
    "font_face": "Microsoft Yahei Mono",
    "font_size": 10.5,
    "gutter": true,
    "highlight_line": true,
    "ignored_packages":
    [
        "Markdown",
        "Vintage"
    ],
    "indent_subsequent_lines": true,
    "line_padding_bottom": 1,
    "line_padding_top": 0,
    "rulers":
    [
        79
    ],
    "sidebar_no_icon": true,
    "sidebar_row_padding_medium": true,
    "sidebar_size_12": true,
    "status_bar_brighter": true,
    "tab_size": 4,
    "tabs_label_not_italic": true,
    "tabs_padding_small": true,
    "tabs_small": true,
    "theme": "Afterglow.sublime-theme",
    "translate_tabs_to_spaces": true,
    "word_wrap": true,
    "wrap_width": 0
}

{% endhighlight %} 

###3. snippet
在tools里，会有新建snippets，snippets是帮助你快速输入一些内容的，比如python的一些开头部分。输入如下内容，之后可以保存到一个新建的snippnets的目录,专门来放这些片段代码,保存的文件后缀必须是"sublime-snippet"。

{% highlight sh %} 
<snippet>
    <content><![CDATA[
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ print_function


def main():


if __name__ == '__main__':
    main()

]]></content>
    <!-- Optional: Set a tabTrigger to define how to trigger the snippet -->
    <tabTrigger>#py</tabTrigger>
    <!-- Optional: Set a scope to limit where the snippet will trigger -->
    <!-- <scope>source.python</scope> -->
</snippet>

{% endhighlight %}

###4. sublimeREPL
可以使用iypthon交互式分析

    {  
    "default_extend_env": {"PATH": "{PATH};C:/Anaconda/Scripts"}  
    }  

也可以用Preferences-Browse Packages...里，把SublimeREPL/config中一些不需要的库都删除掉。另外，对于sublime REPL里的python，可以用刚才Browse Packages的方式打开config中python下的Main.sublime-menu,寻找到`Python - RUN current file`，修改python路径，使得可以进行交互。
{% highlight sh %} 
{"command": "repl_open",
 "caption": "Python - RUN current file",
 "id": "repl_python_run",
 "mnemonic": "d",
 "args": {
     "type": "subprocess",
     "encoding": "utf8",
     "cmd":  ["C:/Anaconda/python", "-u", "-i", "$file_basename"],
     "cwd": "$file_path",
     "syntax": "Packages/Python/Python.tmLanguage",
     "external_id": "python",
     "extend_env": {"PYTHONIOENCODING": "utf-8"}
    }
}
{% endhighlight %}

此外，我们可以给他配置一个快捷键，在Preferences-Key Bindings-User中添加：
{% highlight sh %} 
[
    {"keys":["f5"],
    "caption": "SublimeREPL: Python - RUN current file",
    "command": "run_existing_window_command", "args":
    {
    "id": "repl_python_run",
    "file": "config/Python/Main.sublime-menu"
    }}
]
{% endhighlight %}
###5. ctags
使用ctags进行代码的跳转阅读。
首先在sublime中，安装ctags插件。安装完成之后，会自动的打开一个Package Control Messages，里面会有ctags的一个其他依赖的下载地址和安装方法，比如在windows下，还需要ctags.exe, 可以去[下载](http://prdownloads.sourceforge.net/ctags/ctags58.zip),之后解压到某个文件夹就可以了。

但是，我们想要生成tags，需要把ctags.exe加入系统path里。这样，就可以使用ctags了。比如在某个工程目录下，打开cmd，输入`ctags -R -f .tags` 生成  .tags文件，然后在sublime下就可以用ctrl+t ctrl+t来跳转,用ctrl+t ctrl+b来返回到原来位置。也可以直接在sublime中打开文件夹，之后在sublime中右击文件夹，会有一个`cTags:Rebuild Tags`,点击即可。

###6. 其他插件
* Enhanced-R
* CodeIntel (自动补全)
* Python PEP8Autoformat(python)
* Anaconda (python自动补全以及各种神器)
* SideBarEnhancements 
* Markdown Preview
* GBK Encoding Support 
* CSS Formatter (格式化css)
* fileDiffs (文件对比)

###7. markdown插件
推荐使用Markdown Editing。但是默认的Markdown Editing的颜色不是很帅，不过可以修改。如果是python或者其他类型的也是一样的可以专门的针对这种类型的文件设定自定义的格式。
比如打开一个markdown文件，在Preferences-setting more-Syntax Spacific User，会打开一个Markdown.sublime-sttings，也就是设置Syntax的地方，把下面的代码，复制进去就可以了（当然，比较推荐的方式是把Markdown Editing的一些设置复制过来，我们只是把color_scheme更新一下就可以了）。

{% highlight python %}
{
    "color_scheme": "Packages/Theme - Afterglow/Afterglow-markdown.tmTheme",
    "draw_centered": true,
    "draw_indent_guides": false,
    "trim_trailing_white_space_on_save": false,
    "word_wrap": true,
    "font_size": 11,
    "wrap_width": 80,  // Sets the of characters per line
    "extensions":
    [
        "mdown",
        "md"
    ]
}
{% endhighlight %}

###8. 常用快捷键
- ctrl + -> —— 按照单词移动
- ctrl + d —— 向下选中该单词
- ctrl + shift + up —— 将该行上移
- ctrl + k 、u —— 所选字符转换为大写 
- ctrl + shift + l —— 同时编辑选中的行
- ctrl + shift + d —— 在该行下面复制该行
- esc —— 退出窗口，包括ctrl+f的窗口也可以
对于插件的快捷键，可以通过 Package Settings对应设置。

另外关于 `ctrl + p` 的操作。ctrl+p可以调出窗口，菜单上的解释是gotoanythings ，确实如其所言，调出窗口后，直接输入关键字，可以在已打开的项目文件夹中进行快速文件名导航，而且支持模糊搜索。如果在该窗口里加上` : `前缀即为行跳转(ctrl+G)；如果加上` @ `前缀则是关键字(html里的id，代码里的函数，css里的每条规则，js里则是每个function)导航(ctrl+R)；如果在加上` # `前缀则是定位到具体的变量。

下面是 Sublime Text 3 快捷键精华版

Ctrl+Shift+P：打开命令面板
Ctrl+P：搜索项目中的文件
Ctrl+G：跳转到第几行
Ctrl+W：关闭当前打开文件
Ctrl+Shift+W：关闭所有打开文件
Ctrl+Shift+V：粘贴并格式化
Ctrl+D：选择单词，重复可增加选择下一个相同的单词
Ctrl+L：选择行，重复可依次增加选择下一行
Ctrl+Shift+L：选择多行
Ctrl+Shift+Enter：在当前行前插入新行
Ctrl+X：删除当前行
Ctrl+M：跳转到对应括号
Ctrl+U：软撤销，撤销光标位置
Ctrl+J：选择标签内容
Ctrl+F：查找内容
Ctrl+Shift+F：查找并替换
Ctrl+H：替换
Ctrl+R：前往 method
Ctrl+N：新建窗口
Ctrl+K+B：开关侧栏
Ctrl+Shift+M：选中当前括号内容，重复可选着括号本身
Ctrl+F2：设置/删除标记
Ctrl+/：注释当前行
Ctrl+Shift+/：当前位置插入注释
Ctrl+Alt+/：块注释，并Focus到首行，写注释说明用的
Ctrl+Shift+A：选择当前标签前后，修改标签用的
F11：全屏
Shift+F11：全屏免打扰模式，只编辑当前文件
Alt+F3：选择所有相同的词
Alt+.：闭合标签
Alt+Shift+数字：分屏显示
Alt+数字：切换打开第N个文件
Shift+右键拖动：光标多不，用来更改或插入列内容
鼠标的前进后退键可切换Tab文件
按Ctrl，依次点击或选取，可需要编辑的多个位置
按Ctrl+Shift+上下键，可替换行

**选择类**

Ctrl+D 选中光标所占的文本，继续操作则会选中下一个相同的文本。

Alt+F3 选中文本按下快捷键，即可一次性选择全部的相同文本进行同时编辑。举个栗子：快速选中并更改所有相同的变量名、函数名等。

Ctrl+L 选中整行，继续操作则继续选择下一行，效果和 Shift+↓ 效果一样。

Ctrl+Shift+L 先选中多行，再按下快捷键，会在每行行尾插入光标，即可同时编辑这些行。

Ctrl+Shift+M 选择括号内的内容（继续选择父括号）。举个栗子：快速选中删除函数中的代码，重写函数体代码或重写括号内里的内容。

Ctrl+M 光标移动至括号内结束或开始的位置。

Ctrl+Enter 在下一行插入新行。举个栗子：即使光标不在行尾，也能快速向下插入一行。

Ctrl+Shift+Enter 在上一行插入新行。举个栗子：即使光标不在行首，也能快速向上插入一行。

Ctrl+Shift+[ 选中代码，按下快捷键，折叠代码。

Ctrl+Shift+] 选中代码，按下快捷键，展开代码。

Ctrl+K+0 展开所有折叠代码。

Ctrl+← 向左单位性地移动光标，快速移动光标。

Ctrl+→ 向右单位性地移动光标，快速移动光标。

shift+↑ 向上选中多行。

shift+↓ 向下选中多行。

Shift+← 向左选中文本。

Shift+→ 向右选中文本。

Ctrl+Shift+← 向左单位性地选中文本。

Ctrl+Shift+→ 向右单位性地选中文本。

Ctrl+Shift+↑ 将光标所在行和上一行代码互换（将光标所在行插入到上一行之前）。

Ctrl+Shift+↓ 将光标所在行和下一行代码互换（将光标所在行插入到下一行之后）。

Ctrl+Alt+↑ 向上添加多行光标，可同时编辑多行。

Ctrl+Alt+↓ 向下添加多行光标，可同时编辑多行。

**编辑类**

Ctrl+J 合并选中的多行代码为一行。举个栗子：将多行格式的CSS属性合并为一行。

Ctrl+Shift+D 复制光标所在整行，插入到下一行。

Tab 向右缩进。

Shift+Tab 向左缩进。

Ctrl+K+K 从光标处开始删除代码至行尾。

Ctrl+Shift+K 删除整行。

Ctrl+/ 注释单行。

Ctrl+Shift+/ 注释多行。

Ctrl+K+U 转换大写。

Ctrl+K+L 转换小写。

Ctrl+Z 撤销。

Ctrl+Y 恢复撤销。

Ctrl+U 软撤销，感觉和 Gtrl+Z 一样。

Ctrl+F2 设置书签

Ctrl+T 左右字母互换。

F6 单词检测拼写

**搜索类**

Ctrl+F 打开底部搜索框，查找关键字。

Ctrl+shift+F 在文件夹内查找，与普通编辑器不同的地方是sublime允许添加多个文件夹进行查找，略高端，未研究。

Ctrl+P 打开搜索框。举个栗子：1、输入当前项目中的文件名，快速搜索文件，2、输入@和关键字，查找文件中函数名，3、输入：和数字，跳转到文件中该行代码，4、输入#和关键字，查找变量名。

Ctrl+G 打开搜索框，自动带：，输入数字跳转到该行代码。举个栗子：在页面代码比较长的文件中快速定位。

Ctrl+R 打开搜索框，自动带@，输入关键字，查找文件中的函数名。举个栗子：在函数较多的页面快速查找某个函数。

Ctrl+： 打开搜索框，自动带#，输入关键字，查找文件中的变量名、属性名等。

Ctrl+Shift+P 打开命令框。场景栗子：打开命名框，输入关键字，调用sublime text或插件的功能，例如使用package安装插件。

Esc 退出光标多行选择，退出搜索框，命令框等。

**显示类**

Ctrl+Tab 按文件浏览过的顺序，切换当前窗口的标签页。

Ctrl+PageDown 向左切换当前窗口的标签页。

Ctrl+PageUp 向右切换当前窗口的标签页。

Alt+Shift+1 窗口分屏，恢复默认1屏（非小键盘的数字）

Alt+Shift+2 左右分屏-2列

Alt+Shift+3 左右分屏-3列

Alt+Shift+4 左右分屏-4列

Alt+Shift+5 等分4屏

Alt+Shift+8 垂直分屏-2屏

Alt+Shift+9 垂直分屏-3屏

Ctrl+K+B 开启/关闭侧边栏。

F11 全屏模式

Shift+F11 免打扰模式
