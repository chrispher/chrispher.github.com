---
layout: post
title: The_Analytics_Edge_笔记四篇
category: statistic_analysis
tags: [数据分析]
description: The Analytics Edge课程下的基础数据分析入门、线性回归分析入门、线性回归分析入门、文本分析入门。
---

The Analytics Edge系列的四节笔记。这是第一节：基础数据分析入门，主要是介绍数据分析和R语言。第二节：线性回归分析入门，本节课简单的介绍了线性回归，但不涉及复杂的假设检验和推导公式等。第三节：线性回归分析入门，本节主要简述了logistic回归，同时涉及了混淆矩阵、ROC曲线以及决策边界调整(threshold)的内容。第四节：文本分析入门。本节课主要讲述了介绍了文本分类的问题，涉及一些自然语言处理的基本概念。

<!-- more -->

系列笔记主要根据the analytics Edge by MITx  课件做的笔记。这门课程，主要通过各个案例来学习数据分析方法。

###目录
{:.no_toc}

* 目录
{:toc}

### 第一节:基础数据分析入门

####1、课程简介
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

###第二节:线性回归分析入门

####课堂简要
通过如何建模以判断葡萄酒品质和棒球分析两个案例，介绍了基本的概念，包括回归模型，中间提到了残差、均方差以及p值，但没有涉及推导和数学公式。之后，通过NBA数据，提供了R语言的实现过程。做这篇笔记，核心目的是如何处理回归问题中的一些问题，尤其是如何思考问题——也许是数据分析中，最重要的事情——**常识**。

####案例一、天气预测
- ####背景与问题
考虑到全球变暖，我们想研究一下全球的平均气温与一些变量之间的关系。数据收集自1983年5月到2008年12月([climate_change.csv](https://courses.edx.org/c4x/MITx/15.071x/asset/climate_change.csv)),主要变量如下：  
- Year:观测年
- Month:观测月
- Temp: 全球平均气温(相对于某参考期)
- CO2,N2O,CH4,CFC.11,CFC.12:主要指空气重的各个成分的浓度
- Aerosols: 平流层中气溶胶平均深度(影响光线照射到地球)
- TSI: 太阳辐射总量
- MEL: 多元厄尔尼诺南方涛动指数，衡量厄尔尼诺波动情况
我们现在要通过这些变量来预测温度。
- ####认识数据与数据准备
首先，我们已经提出了问题并且找到了相关的一些数据，希望用这些数据来预测全球平均气温。我们先读取数据:  
`climate = read.csv("climate_change.csv")`;  
通过`str(climate)`,可以看到一共有11个变量，308个观测值，均是数值型数据，之后采用`summary(climate_change)`, 发现数据并没有缺失值。这里，大家可以使用`boxplot`命令，查看各个变量情况。 
之后我们要将数据集划分成训练集合和测试集合。这里请注意**考虑到时间连续性**，我们并没有按照随机选择的方法，按照7:3或8:2进行分割，而是按照时间。  
`train = subset(climate, Year <= 2006)`  
`test = subset(climate, Year > 2006)`
- ####建模
首先我们不采用年份和月份, 使用线性回归:`climatelm = lm(Temp ~ MEI + CO2 + CH4 + N2O + CFC.11 + CFC.12 + TSI + Aerosols, data=train)`  
得到结果不再详细，仅展示：Intercept=-124.6、MEI=0.06421、CO2=0.006457、CH4=0.000124、N2O =-1.653e-02、CFC.11=-0.00663，CFC.12=0.003808，TSI=0.09314，Aerosols=-1.538 。这里，我们根据P值得大小，得知CH4的系数是不显著的。但是我做这个笔记，唯一的目的只是为了：**查看各个系数，我们发现N2O和CFC-11的系数均是负数**，但是我们根据常识知道这两个气体均值温室气体，系数应该是正的才对！因此，我们觉得虽然模型不错，根据残差图也没有觉得有必要去异常值和高杠杆点或是增加高次变量或变量转换。而是基于**常识**判断出存在**多重共线性**。接下来的问题，就是去掉一些不显著变量以及相关变量处理，通过`cor`查看两两之间的相关系数(在笔记<应用线性回归>中提到具体的方法)。
- ####评估与反思
当然，我们可以在建模中，通过主动观察两两之间的相关系数或者通过计算方差膨胀因子等方法，把可能存在的各种潜在问题都去考虑一下，都去探索一下。但是，还是有很多因素和问题，是我们无法通过技术想到或评测到的，比如异常值就一定要删除吗？异常值一定就是可以通过残差图看出来？其实，很多时候，真正决定一个模型好坏的是**业务知识和常识(行业经验和准则)**。之前，一个非常有经验的人告诉我，**做数据分析，永远不要相信自己的分析结果，要有怀疑精神！模型再好，也要用常识去判断，去反思。如果不符合常识，更需要深入的去思考原因**。在很多时候，建立的模型往往因为业务的理解，增加一些或删除一些特征变量而使得结果变好，比如在医疗的癌症判断模型中，因为咨询了资深专家医师，而额外补充一个检测指标，使得最终模型可用。

####案例二、三

- ####离散值处理
使用命令:`pisaTrain$raceeth = relevel(pisaTrain$raceeth, "White")`, 结果是raceeth变量以White为基准的离散化。简单解释下：如果有三个种族黄、白、黑人，以白人为基准，那么离散的结果就是用多个0-1变量来表示：是否是黄种人，是否是黑种人，那么黄白黑分别为(1,0)(0,0),(0,1)。 
- ####变量变换
1.在谷歌流感预测中，我们发现每周访问流感医师数量呈现左偏，之后我们采用了取对数的做法，使得模型效果上升明显。  
2.在时间序列分析中，可以采用前一时刻的值作为输入。关于时间序列分析，在以后会单独讲。
- ####其他
去空值:`pisaTrain = na.omit(pisaTrain)`

###第三节:线性回归分析入门

####课堂简要
本节课主要简述了logistic回归，同时涉及了混淆矩阵、ROC曲线以及决策边界调整(threshold)的内容。在分类过程中，查看混淆矩阵，相比于直接看准确率而言，内容要丰富的多。这里，在R语言使用中，需要一定的技巧，有一定的难度。这里，主要分享一个案例，以及在R语言使用的一些记录。

####案例一 预测贷款偿还
- **背景**
该案例主要分析了贷款人是否会偿还贷款。[数据集](https://courses.edx.org/c4x/MITx/15.071x/asset/loans.csv)来自[LengdingClub.com](https://www.lendingclub.com/info/download-data.action),在数据集中,一共包含了9578个样例、14个变量。主要变量说明如下：  
**credit.policy**: 判断用户是否满足LendingClub.com的信贷承销标准;   
**purpose**: 用户借贷目的，如信用卡、教育、债务整合等;  
**int.rate**: 贷款利率;  
**fico**: FICO指数;  
此外，其他的还包括债务比、过去的贷款记录等等。

- **认识数据**
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

- **建模与分析**
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

####其他案例与感想
在其他案例中，涉及了一些根据P值进行特征选择，以及上面提到的如何灵活的调整threshold来规避风险。当然，这里特别需要注意的是如何根据业务需求来调整threshold。比如在用户流失中，我们可能要尽可能的捕获流失用户，但是考虑到维护成本，我们希望模型尽可能捕获一定会流失的用户，使得维护成本尽可能的低。

###第四节:文本分析入门

####课堂简要
在文本分析中，面临的第一个问题是如何获得数据集。在课堂中，主要是采用了亚马逊的众包平台。而在实际业务中，通常都是需要在产品设计之初，就设定一些评分指标，比如好评、差评等，以有利于后期的发展规划。  
这里需要注意文本的预处理，预处理对最终模型的效果影响是非常非常大的。预处理需要注意：大小写、标点符号、**停用词**。此外，主要注意以下四个处理：
- 有些时候对于网址、数字、特殊字符等有助于我们分类目标的文本进行特殊处理，比如在垃圾邮件分类中，会将所有网址链接转化为一个**自定义单词**‘mailhttps’。当然，在某些自然语言处理包中，还会有**拼写纠正**、词性划分等等。
- 对于stemming（词干提取），课程中提到了一些方法，如设计词库、设计规则等方式。
- 稀疏性，对于一些出现次数特别少的单词，比如稀疏性大于99.5%等词语，需要删掉（即在所有文档中出现次数小于一定次数的单词）。
- 单词权重，tf-idf是比较常用的一种处理方式。  

此外，在IBM案例中，提到了沃森的工作方式：分析问题（问题是寻找什么样的答案）——产生假设（寻找所有可能答案）——评估假设（评估各个假设的置信度）——排序结果（提供支持度最高的答案）。

####案例一
需要注意读取文件时的两个参数stringsAsFactors和encoding="latin1"  
- **read the dataset**  
`tweets = read.csv("tweets.csv", stringsAsFactors=FALSE) `  
`str(tweets)`  
`tweets$Negative = as.factor(tweets$Avg <= -1)`  
`table(tweets$Negative) `   
`library(tm)`  
`library(SnowballC)`  
- **Create corpus, Convert to lower-case, Remove punctuation**   
`corpus = Corpus(VectorSource(tweets$Tweet))`   
`corpus = tm_map(corpus, tolower) `  
`corpus = tm_map(corpus, removePunctuation)`  
- **Remove stopwords and apple, Stem document**  
`stopwords("english")[1:10]`  
`corpus = tm_map(corpus, removeWords, c("apple", stopwords("english"))) `  
`corpus = tm_map(corpus, stemDocument) `  
- **frequencies, sparsity**  
`frequencies = DocumentTermMatrix(corpus) `  
`inspect(frequencies[1000:1005,505:515]) `  
`findFreqTerms(frequencies, lowfreq=20)`  
`sparse = removeSparseTerms(frequencies, 0.995) # Remove sparse terms`  
Convert to a data frame  
`tweetsSparse = as.data.frame(as.matrix(sparse))`  
Make all variable names R-friendly and Add dependent variable   
`colnames(tweetsSparse) = make.names(colnames(tweetsSparse))`  
`tweetsSparse$Negative = tweets$Negative`
- **Split the data**  
`library(caTools)`  
`set.seed(123)`  
`split = sample.split(tweetsSparse$Negative, SplitRatio = 0.7)`  
`trainSparse = subset(tweetsSparse, split==TRUE)`  
`testSparse = subset(tweetsSparse, split==FALSE)`  
之后，根据得到的数据，建模即可。