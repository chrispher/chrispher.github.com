---
layout: post
title: TAE：基础数据分析入门
category: 统计分析
tags: [TAE, 数据分析]
description: The Analytics Edge课程下的基础数据分析入门。
---

The Analytics Edge系列的四节笔记，这是第一节：基础数据分析入门，主要是介绍数据分析和R语言。。

<!-- more -->

系列笔记主要根据the analytics Edge by MITx  课件做的笔记。这门课程，主要通过各个案例来学习数据分析方法。

###目录
{:.no_toc}

* 目录
{:toc}

### 1、课程简介
- **什么是分析**  
分析就是科学的使用**数据**(数据包括文本、语音等等)来建立**模型**，使得个人、企业或组织能够得到更好的**决策**并增加**价值**。
- **关于这门课**  
1）主要信息：分析学为个人、企业和组织提供了竞争优势，是企业取得成功的关键。  
2）方法论：本课程通过实际生活中的案例和数据来讲述分析技术。  
3）目标：让你相信分析学的优势，让你能够在你的职业和生活中使用分析学。
- **案例总结**  
课程将会讲述IBM的智能问答机器人Watson，在线婚恋、心脏研究、公司收益管理、放射治疗、Twitter、Netflix等等。

### 2、R语言
- **变量**：变量名称，名称要有意义，一般会使用Student.Count，而非Student_Count
- **导入数据**：load，read.csv等
- **基本统计分析**：查看变量名，str,summary函数
- **合并数据**：cbine，rbine分别是按照列和行合并  

### 3、芝加哥案例
本案例是针对美国芝加哥暴力犯罪和财产犯罪数据，芝加哥是美国第三大城市，人口超过270万。在本次讨论中，主要针对财产犯罪(“property crime”)中的“motor vehicle theft”，即偷窃或尝试偷窃汽车的行为。数据来自[mvtWeek1](https://courses.edx.org/c4x/MITx/15.071x/asset/mvtWeek1.csv)。
#### 数据说明
- **ID**：每个观测值的标识
- **Date**：犯罪日期
- **LocationDescription**：犯罪发生地点
- **Arest**：是否进行拘捕
- **Domestic**：是否是家庭犯罪，即犯罪是针对家庭内部成员
- **Beat**：犯罪发生小区，该地区是芝加哥政府细分的最小地区
- **District**：犯罪发生的行政区，每个行政区包括很多小区
- **CommunityArea**：犯罪发生的社区，芝加哥划分了77个社区
- **Year**：犯罪发生的年份
- **Latitude**：犯罪发生地点的纬度
- **Longitude**：犯罪发生的经度

#### 数据探索
- **基本描述**  
我们观测到191641的观测值，11个变量。  
```mvt = read.csv('mvtWeek1.csv')  #读取数据```  
```str(mvt)  #查看变量```  
```table(mvt$Arrest) #犯罪被拘捕和未拘捕的数量```
- **时间转化**  
R在读取数据的时候，不能直接读取为时间，需要进一步的转化。  
```DateConvert = as.Date(strptime(mvt$Date, "%m/%d/%y %H:%M")) #把文本转化成日期对象```  
```mvt$Month = months(DateConvert) #获得月份```  
此外，最终被拘捕的犯罪量最多的月份是哪个月？  
```table(mvt$Month,mvt$Arrest) #观察这个table就可知,第二列是True```  
``` which.max(table(mvt$Month,mvt$Arrest)[,2])```
- **可视化**  
熟悉常用的几个可视化命令,hist、boxplot、plot等。  
```boxplot(mvt$Year~mvt$Arrest) # 查看拘捕与否的时间分布，07年以后明显较少```  
这里提到一下求和，可以按照列或者行求和，rowSums，colSums。如果我们想得到不同年份的被拘捕与否的比例，我们可以用```table(mvt$Year,mvt$Arrest) / rowSums(table(mvt$Year,mvt$Arrest))```  
此外，我们提问：犯罪率如何变化呢？提示采用plot绘制出每年犯罪率即可。
- **最恶劣城市**  
在这里，我们通过sort+table命令，实现犯罪次数最多的城市。  
```Location.sort = sort(table(mvt$LocationDescription),decreasing=T)```  
我们可以通过```names```命令看到具体的城市名。用```subset```命令进行选择。这里注意，我们可以使用```&, |, ! ```来实现与或非。
- **其他**  
1) 这里需要注意一些其他常用命令。课堂上着重强调了```tapply(argument1, argument2, argument3, na.rm=TRUE)``` 意思是 Group argument 1 by argument 2 and apply argument 3，注意缺失值。   
2) **summary**命令也可以看到各个属性的缺失值情况；而table在计算中自动忽略缺失值；  
3) 使用命令**jitter**，能够在原始数据上增加一定的随机噪声。  
4）在使用merge的时候，主要映射条件，如```CPS = merge(CPS, MAC, by.x="MetroAreaCode", by.y="Code", all.x=TRUE)```，这里all.x=TRUE，即选择按照哪个数据集完整匹配。  
5）巧妙使用均值和is.na：```tapply(is.na(CPS$MetroAreaCode),CPS$State,mean)```,计算MetroAreaCode(二元)是否为空下的各个State比例。类似的也有```sort(tapply(CPS$Race == "Asian", CPS$MetroArea, mean))```等。

### 4、总结
此外，我们还有“股票数据”等练习，在此不一一笔记。在其他练习中，主要按照：**读取数据** ——> **评估缺失值** ——> **summary数据** ——> **table数据** ——>**plot数据**。这里着重强调一下_**缺失值**_,需要注意缺失值的数据，尤其是考虑缺失值是否存在某在模式，比如我们通过```table(CPS$Sex, is.na(CPS$Married))```来判断Married是否缺失对Sex的影响（简单而言，看Married是否缺失下的男女比例是否变化，正式的可以采用卡方检验）。   
在这个练习过程中，进一步的强化R语言的使用，同时我们也看到在数据探索阶段的一些常用方法——如何查看各个样本分布。在其他的练习中，除了样本分布之外，还要求我们计算均值、方差、中位数等基本统计量。这些都是关于基本的**描述统计分析**，也是探索数据，而在过程中**可视化**无疑是最有利于我们的。
