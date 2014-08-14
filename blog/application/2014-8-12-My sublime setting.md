My sublime text2 setting
============================
windows下sublime text2的一些设置和调整

[TOC]

####1. 安装插件Package Control 
在控制台输入以下代码后重启：

    import urllib2,os; pf='Package Control.sublime-package';
    ipp = sublime.installed_packages_path();
    os.makedirs( ipp ) if not os.path.exists(ipp) else None;
    urllib2.install_opener( urllib2.build_opener( urllib2.ProxyHandler( )));
    open( os.path.join( ipp, pf), 'wb' ).write( urllib2.urlopen( 'http://sublime.wbond.net/' +pf.replace( ' ','%20' )).read());
    print( 'Please restart Sublime Text to finish installation')

####2. 推荐主题theme fatland
主题设置

    {
        "color_scheme": "Packages/Theme - Flatland/Flatland Dark.tmTheme",
        "detect_indentation": true,
        "draw_centered": false,
        "fold_buttons": true,
        "font_face": "Consolas",
        "font_size": 11,
        "gutter": true,
        "highlight_line": true,
        "ignored_packages":
        [
            "Vintage"
        ],
        "indent_subsequent_lines": true,
        "line_padding_bottom": 1,
        "line_padding_top": 0,
        "rulers":
        [
            100
        ],
        "theme": "Flatland Dark.sublime-theme",
        "word_wrap": false,
        "wrap_width": 0
    }


####3. Enhanced-R  
build R 

    {
        "cmd": ["Rscript.exe", "$file"],
        "path": "C:\\Program Files\\R\\R-3.0.3\\bin\\i386\\",
        "selector": "source.r"
    }


####4. sublimeREPL 
可以使用iypthon交互式分析

    {  
    "default_extend_env": {"PATH": "{PATH};C:\\Program Files\\R\\R-3.0.3\\bin\\i386"}  
    }  

####5. 其他插件
* CodeIntel  
* SideBarEnhancements 
* Markdown Preview
* Python PEP8 Autoformat
* GBK Encoding Support

####6. 修改tab键为4个空格
    preferences - Settings-Default:  
    "tab_size": 4  
    "translate_tabs_to_spaces": true  

####7. 常用快捷键
- ctrl + -> —— 按照单词移动
- ctrl + d —— 向下选中该单词
- ctrl + shift + up —— 将该行上移
- ctrl + k 、u —— 所选字符转换为大写 
- ctrl + shift + l —— 同时编辑选中的行
- ctrl + shift + d —— 在该行下面复制该行
对于插件的快捷键，可以通过 Package Settings对应设置。
