---
layout: post
title: VS2010环境下编译cuda-convnet(Win7_64位)
category: Application
tags: [convnet, CNN, win7, 配置, 深度学习]
---

系统是win7 64位, 已经提前安装了Visual Studio2010， CUDA5.5，Python安装的是Anaconda环境(可以自行搜索安装)；

注：文章主要参考chenli2010的[blog](http://blog.csdn.net/chenli2010/article/details/17253759)

<!-- more -->

### 目录
- [编译过程](#编译过程)
- [测试过程](#测试过程)
- [常见错误](#常见错误)

<a name="编译过程"/>

### 编译过程

- 0).准备工作

    安装Visual Studio2010, Anaconda, CUDA5.5 SDK；
    下载需要使用的[cuda-convnet: https://github.com/dnouri/cuda-convnet](https://github.com/dnouri/cuda-convnet)；
    下载cuda-convnet的VS2010工程：[cuda-convnet-vs-proj.zip](https://code.google.com/p/cuda-convnet/downloads/detail?name=cuda-convnet-vs-proj.zip)；
    下载已经编译好了的openBLAS and pthread-x64库：[LIBS.zip](https://www.dropbox.com/s/obyzzcankkknjip/LIBS.zip)。相关工具，放在[百度网盘](http://pan.baidu.com/s/1bn3wxdX)里了(包括测试过程中的数据等文件)；

- 1).解压cuda-convnet-vs-proj.zip

	用记事本/写字板打开pyconvnet.vcxproj；将其中的CUDA 4.0 改成CUDA 5.5，总共有两处地方要修改。当然，如果您的CUDA SDK版本为5.0，那就把这里的数字改成5.0就行了。

- 2).解压cuda-convnet的源码：cuda-convnet-master.zip，将其中的include和src文件夹拷贝到工程文件夹中。

- 3).双击convnet.sln打开工程，将工程设置为x64 release模式。

- 4).在include/nvmatrix/nvmatrix.cuh中加入`#include <pthread.h>`。

- 5).解压LIBS.zip，将解压后的文件夹（LIBS）拷贝到工程文件夹中。

- 6).右击工程: 属性(Property) -> 配置属性(Configuration Properties)  -> C/C++ -> 常规(General) -> 附加包含目录(Additional Include Directories)：删除所有，把下面的复制过去；需要注意Python的路径和CUDA的路径；

    `C:/Anaconda/Lib/site-packages/numpy/core/include/numpy;`  
    `C:/Anaconda/include;./include/common;./include/nvmatrix;`  
    `./include/cudaconv2;./include;$(CudaToolkitIncludeDir);`  
    `./;./LIBS/Pre-built.2/include;./LIBS/include;`  
    `C:/NVIDIA/CUDA/CUDASamples/common/inc;`  

- 7).右击工程: 属性(Property) -> 配置属性(Configuration Properties)  -> 链接器(Linker)  -> 常规(General) -> 附加包含目录(Additional Include Directories)：加入库目录。最终如下所示：

    ` C:/Anaconda/libs;C:/NVIDIA/CUDA/CUDAToolkit/lib/x64;`  
    `./LIBS;./LIBS/Pre-built.2/lib;$(CudaToolkitLibDir); `

- 8).右击工程: 属性(Property) -> 配置属性(Configuration Properties)  -> 链接器(Linker) -> 输入(Input) -> 附加依赖项(Additional Dependencies)：加入附加依赖项。保持原有的不变，在其中加入python27.lib;libopenblas.lib。

- 9).右击工程: 属性(Property) -> 配置属性(Configuration Properties)  -> C/C++ -> 预处理器 -> 预处理器定义(Preprocessor Definitions) 中删除USE_MKL;

- 10).右击工程: 属性(Property) -> 配置属性(Configuration Properties) -> CUDA C/C++ -> Host -> 预处理器定义(Preprocessor Definitions) 中删除USE_MKL;

- 11).右击工程: 属性(Property) -> 配置属性(Configuration Properties) ->  常规(General) -> 目标文件名(Target Name): 改为 _convnet

- 12).build整个工程。如果编译成功，就OK了。


<a name="测试过程"/>

### 测试过程
测试过程遵循cuda-convnet官方主页的说明文档步骤

- 0).下载数据集：[cifar-10-py-colmajor.tar.gz](http://www.cs.toronto.edu/~kriz/cifar-10-py-colmajor.tar.gz) 。

- 1).在工程文件夹中新建一个data文件夹，将cifar-10-py-colmajor.tar.gz 解压后拷贝到data中。

- 2).在工程文件夹中新建一个tmp文件夹。

- 3).将cuda-convnet-master中除了include和scr之外的所有文件都拷贝到工程文件夹中。

- 4).下载 [Dependency Walker](http://www.dependencywalker.com)，将工程文件夹中的pyconvnet.pyd（新版名称为_convnet.pyd）在Dependency Walker中打开，检查缺少的dll文件。实际上，按照上面的步骤几乎肯定缺少两个dll文件：libopenblas.dll 和pthreadVC2_x64.dll，这两个文件都在LIBS文件夹中。将它们拷贝到工程文件夹即可。可能还会缺少其他的dll，比如，我的机器就缺少 libgfortran-3.dll和ieshims.dll，需要从网上下载。

- 5).在cmd中运行命令：

`python convnet.py --data-path=./data/cifar-10-py-colmajor --save-path=./tmp --test-range=6 --train-range=1-5 --layer-def=./example-layers/layers-19pct.cfg --layer-params=./example-layers/layer-params-19pct.cfg --data-provider=cifar --test-freq=13`

运行成功会出现如下信息：

<img src="/images/deeplearning/convnet_test.png" height="100%" width="100%">

<a name="常见错误"/>

### 常见错误

- 1).测试阶段，错误提示为：

    `import pyconvnet `  
    `ImportError: DLL load failed: 找不到指定的模块。`

    解决方法：这种错误是缺少dll引起的，用Dependency Walker进行全面检查。

- 2).测试阶段，错误提示为：

	` Error at C:/cuda-convnet-vs-proj/src/nvmatrix.cu:276`

    解决方法：这种错误是由于系统无法调用GPU，请运行CUDA SDK的例程，确认GPU是否可用。
    注：如果用户是通过远程桌面连接到Window 7，那么他是无法调用GPU的，这时，可以通过VNC或者向日葵远程控制软件来远程登录Win 7系统。

- 3).测试阶段，出现错误：

	` No Module named _convnet`

   这个是错误是目标名称不对应导致，可以右击工程: 属性(Property) -> 配置属性(Configuration Properties) ->  常规(General) -> 目标文件名(Target Name): 改为 _convnet