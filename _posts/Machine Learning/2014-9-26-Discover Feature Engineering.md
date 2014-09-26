---
layout: post
title: Discover Feature Engineering
comments: true
category: Machine Learning
tags: [Feature, ]
---

Note: This article originates from the [network](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

### Contents
<!-- MarkdownTOC depth=4 -->
- [1.概述](#1.概述)
- [2.深度学习的过拟合](#2.深度学习的过拟合)
- [3.dropout](#3.dropout)
- [4. Appendix](#4. Appendix)
<!-- /MarkdownTOC -->


<a name="1.Introduction" />

###1.Introduction
Feature engineering is an informal topic, but one that is absolutely known and agreed to be key to success in applied machine learning.

In creating this guide I went wide and deep and synthesized all of the material I could.You will discover what feature engineering is, what problem it solves, why it matters, how to engineer features, who is doing it well and where you can go to learn more and get good at it.

If you read one article on feature engineering, I want it to be this one.

> feature engineering is another topic which doesn’t seem to merit any review papers or books, or even chapters in books, but it is absolutely vital to ML success. [...] Much of the success of machine learning is actually success in engineering features that a learner can understand.  — Scott Locklin, in “[Neglected machine learning ideas](https://scottlocklin.wordpress.com/2014/07/22/neglected-machine-learning-ideas/)”

<a name="2.Problem that Feature Engineering Solves" />

###2.Problem that Feature Engineering Solves
When your goal is to get the best possible results from a predictive model, you need to get the most from what you have. This includes getting the best results from the algorithms you are using. It also involves getting the most out of the data for your algorithms to work with.

**How do you get the most out of your data for predictive modeling?**

This is the problem that the process and practice of feature engineering solves.

> Actually the success of all Machine Learning algorithms depends on how you present the data. — Mohammad Pezeshki, answer to “[What are some general tips on feature selection and engineering that every data scientist should know?](http://www.quora.com/What-are-some-general-tips-on-feature-selection-and-engineering-that-every-data-scientist-should-know)”

<a name="3.Importance of Feature Engineering"/>

###3.Importance of Feature Engineering

The features in your data will directly influence the predictive models you use and the results you can achieve.You can say that: the better the features that you prepare and choose, the better the results you will achieve. It is true, but it also misleading.

The results you achieve are a factor of the model you choose, the data you have available and the features you prepared. Even your framing of the problem and objective measures you’re using to estimate accuracy play a part. Your results are dependent on many inter-dependent properties.

You need great features that describe the structures inherent in your data.

- **Better features means flexibility.**
You can choose “the wrong models” (less than optimal) and still get good results. Most models can pick up on good structure in data. The flexibility of good features will allow you to use less complex models that are faster to run, easier to understand and easier to maintain. This is very desirable.

- **Better features means simpler models.**
With well engineered features, you can choose “the wrong parameters” (less than optimal) and still get good results, for much the same reasons. You do not need to work as hard to pick the right models and the most optimized parameters.

With good features, you are closer to the underlying problem and a representation of all the data you have available and could use to best characterize that underlying problem. 

- **Better features means better results.**
> The algorithms we used are very standard for Kagglers. [...]  We spent most of our efforts in feature engineering.  — Xavier Conort, on “[Q&A with Xavier Conort](http://blog.kaggle.com/2013/04/10/qa-with-xavier-conort/)” on winning the Flight Quest challenge on Kaggle

<a name="4.What is Feature Engineering?"/>

###4.What is Feature Engineering?
Here is how I define feature engineering:

> **Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.**

You can see the dependencies in this definition:
- The performance measures you’ve chosen (RMSE? AUC?)
- The framing of the problem (classification? regression?)
- The predictive models you’re using (SVM?)
- The raw data you have selected and prepared (samples? formatting? cleaning?)

> feature engineering is manually designing what the input x’s should be. — Tomasz Malisiewicz, answer to “[What is feature engineering?](http://www.quora.com/What-is-feature-engineering)”

####4.1 Feature Engineering is a Representation Problem

Machine learning algorithms learn a solution to a problem from sample data.In this context, feature engineering asks: what is the best representation of the sample data to learn a solution to your problem?

It’s deep. Doing well in machine learning, even in artificial intelligence in general comes back to representation problems. It’s hard stuff, perhaps unknowable (or at best intractable) to know the best representation to use, a priori.

> you have to turn your inputs into things the algorithm can understand. — Shayne Miel, answer to “[What is the intuitive explanation of feature engineering in machine learning?](http://www.quora.com/What-is-the-intuitive-explanation-of-feature-engineering-in-machine-learning)”

####4.2 Feature Engineering is an Art
It is an art like engineering is an art, like programming is an art, like medicine is an art.There are well defined procedures that are methodical, provable and understood.

The data is a variable and is different every time. You get good at deciding which procedures to use and when, by practice. By empirical apprenticeship. Like engineering, like programming, like medicine, like machine learning in general.

Mastery of feature engineering comes with hands on practice, and study of what others that are doing well are practicing.

> …some machine learning projects succeed and some fail. What makes the difference? Easily the most important factor is the features used. — Pedro Domingos, in “[A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)” (PDF)

<a name="5. Sub-Problems of Feature Engineering"/>

###5. Sub-Problems of Feature Engineering
It is common to think of feature engineering as one thing. For example, for a long time for me, feature engineering was feature construction.

I would think to myself “I’m doing feature engineering now” and I would pursue the question “How can I decompose or aggregate raw data to better describe the underlying problem?” The goal was right, but the approach was one of a many.

In this section we look at these many approaches and the specific sub-problems that they are intended to address. Each could be an in depth article of their own as they are large and important areas of practice and study.

<a name="5.1 Feature"/>

####5.1 Feature: An attribute useful for your modeling task
Let’s start with data and what is a [feature](http://en.wikipedia.org/wiki/Feature_(machine_learning)).

Tabular data is described in terms of observations or instances (rows) that are made up of variables or attributes (columns). An attribute could be a feature.

The idea of a feature, separate from an attribute, makes more sense in the context of a problem. A feature is an attribute that is useful or meaningful to your problem. It is an important part of an observation for learning about the structure of the problem that is being modeled.

I use “meaningful” to discriminate attributes from features. Some might not. I think there is no such thing as a non-meaningful feature. If a feature has no impact on the problem, it is not part of the problem.

In computer vision, an image is an observation, but a feature could be a line in the image. In natural language processing, a document or a tweet could be an observation, and a phrase or word count could be a feature. In speech recognition, an utterance could be an observation, but a feature might be a single word or phoneme.

<a name="5.2 Feature Importance"/>

####5.2 Feature Importance: An estimate of the usefulness of a feature
You can objectively estimate the usefulness of features.

This can be helpful as a pre-cursor to selecting features. Features are allocated scores and can then be ranked by their scores. Those features with the highest scores can be selected for inclusion in the training dataset, whereas those remaining can be ignored.

Feature importance scores can also provide you with information that you can use to extract or construct new features, similar but different to those that have been estimated to be useful.

A feature may be important if it is highly correlated with the dependent variable (the thing being predicted). Correlation coefficients and other univariate (each attribute is considered independently) methods are common methods.

More complex predictive modeling algorithms perform feature importance and selection internally while constructing their model. Some examples include MARS, Random Forest and Gradient Boosted Machines. These models can also report on the variable importance determined during the model preparation process.

<a name="5.3 Feature Extraction"/>

####5.3 Feature Extraction: The automatic construction of new features from raw data

Some observations are far too voluminous in their raw state to be modeled by predictive modeling algorithms directly.

Common examples include image, audio, and textual data, but could just as easily include tabular data with millions of attributes.

[Feature extraction](http://en.wikipedia.org/wiki/Feature_extraction) is a process of automatically reducing the dimensionality of these types of observations into a much smaller set that can be modelled.

For tabular data, this might include projection methods like Principal Component Analysis and unsupervised clustering methods. For image data, this might include line or edge detection. Depending on the domain, image, video and audio observations lend themselves to many of the same types of DSP methods.

Key to feature extraction is that the methods are automatic (although may need to be designed and constructed from simpler methods) and solve the problem of unmanageably high dimensional data, most typically used for analog observations stored in digital formats.

<a name="5.4 Feature Selection"/>

####5.4 Feature Selection: From many features to a few that are useful
Not all features are created equal.

Those attributes that are irrelevant to the problem need to be removed. There will be some features that will be more important than others to the model accuracy. There will also be features that will be redundant in the context of other features.

Feature selection addresses these problems by automatically selecting a subset that are most useful to the problem.

[Feature selection](http://en.wikipedia.org/wiki/Feature_selection) algorithms may use a scoring method to rank and choose features, such as correlation or other feature importance methods.

More advanced methods may search subsets of features by trial and error, creating and evaluating models automatically in pursuit of the objectively most predictive sub-group of features.

There are also methods that bake in feature selection or get it as a side effect of the model. Stepwise regression is an example of an algorithm that automatically performs feature selection as part of the model construction process.

Regularization methods like LASSO and ridge regression may also be considered algorithms with feature selection baked in, as they actively seek to remove or discount the contribution of features as part of the model building process.

<a name="5.5 Feature Construction"/>

####5.5 Feature Construction: The manual construction of new features from raw data
The best results come down to you, the practitioner, crafting the features.

Feature importance and selection can inform you about the objective utility of features, but those features have to come from somewhere.

You need to manually create them. This requires spending a lot of time with actual sample data (not aggregates) and thinking about the underlying form of the problem, structures in the data and how best to expose them to predictive modeling algorithms.

With tabular data, it often means a mixture of aggregating or combining features to create new features, and decomposing or splitting features to create new features.

With textual data, it often means devising document or context specific indicators relevant to the problem. With image data, it can often mean enormous amounts of time prescribing automatic filters to pick out relevant structures.

This is the part of feature engineering that is often talked the most about as an artform, the part that is attributed the importance and signalled as the differentiator in competitive machine learning.

It is manual, it is slow, it requires lots of human brain power, and it makes a big difference.

> Feature engineering and feature selection are not mutually exclusive.  They are both useful.  I’d say feature engineering is more important though, especially because you can’t really automate it. — Robert Neuhaus, answer to “[Which do you think improves accuracy more, feature selection or feature engineering?](http://www.quora.com/How-valuable-do-you-think-feature-selection-is-in-machine-learning-Which-do-you-think-improves-accuracy-more-feature-selection-or-feature-engineering)”

<a name="5.6 Feature Learning"/>

####5.6 Feature Learning: The automatic identification and use of features in raw data
Can we avoid the manual load of prescribing how to construct or extract features from raw data?

Representation learning or [feature learning](http://en.wikipedia.org/wiki/Feature_learning) is an effort towards this goal.

Modern deep learning methods are achieving some success in this area, such as autoencoders and restricted Boltzmann machines. They have been shown to automatically and in a unsupervised or semi-supervised way, learn abstract representations of features (a compressed form), that in turn have supported state-of-the-art results in domains such as speech recognition, image classification, object recognition and other areas.

We do not have automatic feature extraction or construction, yet, and we will probably never have automatic feature engineering.

The abstract representations are prepared automatically, but you cannot understand and leverage what has been learned, other than in a black-box manner. They cannot (yet, or easily) inform you and the process on how to create more similar and different features like those that are doing well, on a given problem or on similar problems in the future. The acquired skill is trapped.

Nevertheless, it’s fascinating, exciting and an important and modern part of feature engineering.

<a name="6. Process of Feature Engineering"/>

###6. Process of Feature Engineering
Feature engineering is best understood in the broader process of applied machine learning.

You need this context.

<a name="6.1 Process of Machine Learning"/>

####6.1 Process of Machine Learning
The process of applied machine learning (for lack of a better name) that in a broad brush sense involves lots of activities. Up front is problem definition, next is  data selection and preparation, in the middle is model preparation, evaluation and tuning and at the end is the presentation of results.

Process descriptions like [data mining and KDD](http://machinelearningmastery.com/what-is-data-mining-and-kdd/) help to better understand the tasks and subtasks. You can pick and choose and phrase the process the way you like. I’ve talked a lot about [this before](http://machinelearningmastery.com/process-for-working-through-machine-learning-problems/).

A picture relevant to our discussion on feature engineering is the front-middle of this process. It might look something like the following:

- (tasks before here…)
- **Select Data**: Integrate data, de-normalize it into a dataset, collect it together.
- **Preprocess Data**: Format it, clean it, sample it so you can work with it.
- **Transform Data**: Feature Engineer happens here.
- **Model Data**: Create models, evaluate them and tune them.
- (tasks after here…)

The traditional idea of “Transforming Data” from a raw state to a state suitable for modeling is where feature engineering fits in. Transform data and feature engineering may in fact be synonyms.

This picture helps in a few ways.

You can see that before feature engineering, we are munging out data into a format we can even look at, and just before that we are collating and denormalizing data from databases into some kind of central picture.

We can, and should go back through these steps as we identify new perspectives on the data.

For example, we may have an attribute that is an aggregate field, like a sum. Rather than a single sum, we may decide to create features to describe the quantity by time interval, such as season. We need to step backward in the process through Preprocessing and even Selecting data to get access to the “real raw data” and create this feature.

We can see that feature engineering is followed by modeling.

It suggests a strong interaction with modeling, reminding us of the interplay of devising features and testing them against the coalface of our test harness and final performance measures.

This also suggests we may need to leave the data in a form suitable for the chosen modeling algorithm, such as normalize or standardize the features as a final step. This sounds like a preprocessing step, it probably is, but it helps us consider what types of finishing touches are needed to the data before effective modeling.


<a name="6.2 Iterative Process of Feature Engineering"/>

####6.2 Iterative Process of Feature Engineering

**"TO---DO--"**