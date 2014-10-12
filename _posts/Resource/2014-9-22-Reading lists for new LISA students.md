---
layout: post
title: Reading lists for new LISA students
comments: true
category: Resource
tags: [Deep Learning, Reading, lists]
---

Benjio给LISA学生的[Reading lists](https://docs.google.com/document/d/1IXF3h0RU5zz4ukmTrVKVotPQypChscNGf5k6E25HGvA/edit)

###content
- [Research in General](#research-in-general)
- [Basics of machine learning](#basics-of-machine-learning)
- [Basics of deep learning](#basics-of-deep-learning)
- [Feedforward nets](#feedforward-nets)
- [MCMC](#mcmc)
- [Restricted Boltzmann Machines](#restricted-boltzmann-machines)
- [Boltzmann Machines](#boltzmann-machines)
- [Regularized Auto-Encoders](#regularized-auto-encoders)
- [Regularization](#regularization)
- [Stochastic Nets & GSNs](#stochastic-nets-&-gsns)
- [Others](#others)
- [Recurrent Nets](#recurrent-nets)
- [Convolutional Nets](#convolutional-nets)
- [Optimization issues with DL](#optimization-issues-with-dl)
- [NLP + DL](#nlp+dl)
- [CV+RBM](#cv+rbm)
- [CV + DL](#cv-+-dl)
- [Scaling Up](#scaling up)
- [DL + Reinforcement learning](#dl+reinforcement-learning)
- [Graphical Models Background](#graphical-models-background)
- [Writing](#writing)
- [Software documentation](#software-documentation)
- [Software lists of built-in commands/functions](#software-lists-of-built-in-commands/functions)
- [Other Software stuff to know about:](#other-software-stuff-to-know-about)


Reading lists for new LISA students

####Research in General
- [How to write a great research paper](https://research.microsoft.com/en-us/um/people/simonpj/papers/giving-a-talk/writing-a-paper-slides.pdf)

####Basics of machine learning

- [bengioy/DLbook/math.html](http://www.iro.umontreal.ca/~bengioy/DLbook/math.html)
- [bengioy/DLbook/ml.html](http://www.iro.umontreal.ca/~bengioy/DLbook/ml.html)

####Basics of deep learning
- [bengioy/DLbook/intro.html](http://www.iro.umontreal.ca/~bengioy/DLbook/intro.html)
- [bengioy/DLbook/mlp.html](http://www.iro.umontreal.ca/~bengioy/DLbook/mlp.html)
- [Learning deep architectures for AI](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf)
- [Practical recommendations for gradient-based training of deep architectures](http://arxiv.org/abs/1206.5533)
- [Quick’n’dirty introduction to deep learning: Advances in Deep Learning](http://users.ics.aalto.fi/kcho/papers/dissertation_draft.pdf) 
- [A fast learning algorithm for deep belief nets](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
- [Greedy Layer-Wise Training of Deep Networks](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_739.pdf)
- [Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- [Contractive auto-encoders: Explicit invariance during feature extraction](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf)
- [Why does unsupervised pre-training help deep learning?](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf)
- [An Analysis of Single Layer Networks in Unsupervised Feature Learning](http://web.eecs.umich.edu/~honglak/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf)
- [The importance of Encoding Versus Training With Sparse Coding and Vector Quantization](http://www.stanford.edu/~acoates/papers/coatesng_icml_2011.pdf)
- [Representation Learning: A Review and New Perspectives ](https://arxiv.org/abs/1206.5538)
- [Deep Learning of Representations: Looking Forward ](https://arxiv.org/abs/1305.0445)
- [Measuring Invariances in Deep Networks](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2009_0463.pdf)
- [Neural networks course at USherbrooke](http://info.usherbrooke.ca/hlarochelle/cours/ift725_A2013/contenu.html) [youtube]](http://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)

####Feedforward nets
- [bengioy/DLbook/mlp.html](http://www.iro.umontreal.ca/~bengioy/DLbook/mlp.html)
- [“Improving Neural Nets with Dropout” by Nitish Srivastava](http://www.cs.toronto.edu/~nitish/msc_thesis.pdf)
- [“Fast Drop Out” ](http://nlp.stanford.edu/pubs/sidaw13fast.pdf)
- [“Deep Sparse Rectifier Neural Networks”](http://deeplearningworkshopnips2010.files.wordpress.com/2010/11/nipswrkshp2010-cameraready.pdf)
- [“What is the best multi-stage architecture for object recognition?”](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)
- [“Maxout Networks”](http://arxiv.org/pdf/1302.4389v4.pdf)

####MCMC
- [Iain Murray’s MLSS slides](http://mlg.eng.cam.ac.uk/mlss09/mlss_slides/Murray_1.pdf)
- [Radford Neal’s Review Paper](http://www.cs.toronto.edu/pub/radford/review.pdf) (old but still very comprehensive)
- [Better Mixing via Deep Representations](https://arxiv.org/abs/1207.4404)

####Restricted Boltzmann Machines
- [Unsupervised learning of distributions of binary vectors using 2-layer networks](http://cseweb.ucsd.edu/~yfreund/papers/freund94unsupervised.pdf)
- [A practical guide to training restricted Boltzmann machines](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
- [Training restricted Boltzmann machines using approximations to the likelihood gradient](http://icml2008.cs.helsinki.fi/papers/638.pdf)
- [Tempered Markov Chain Monte Carlo for training of Restricted Boltzmann Machine](http://www.iro.umontreal.ca/~lisa/pointeurs/tempered_tech_report2009.pdf)
- [How to Center Binary Restricted Boltzmann Machines](http://arxiv.org/pdf/1311.1354v1.pdf)
- [Enhanced Gradient for Training Restricted Boltzmann Machines](http://users.ics.aalto.fi/kcho/papers/nc13rbm.pdf)
- [Using fast weights to improve persistent contrastive divergence](http://www.cs.toronto.edu/~tijmen/fpcd/fpcd.pdf)
- [Training Products of Experts by Minimizing Contrastive Divergence](http://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)

####Boltzmann Machines
- [Deep Boltzmann Machines (Salakhutdinov & Hinton)](http://www.cs.toronto.edu/~hinton/absps/dbm.pdf)
- [Multimodal Learning with Deep Boltzmann Machines](http://www.cs.toronto.edu/~rsalakhu/papers/Multimodal_DBM.pdf)
- [Multi-Prediction Deep Boltzmann Machines ](http://papers.nips.cc/paper/5024-multi-prediction-deep-boltzmann-machines)
- [A Two-stage Pretraining Algorithm for Deep Boltzmann Machines](http://users.ics.aalto.fi/kcho/papers/nips12workshop.pdf)

####Regularized Auto-Encoders
- [The Manifold Tangent Classifier](http://books.nips.cc/papers/files/nips24/NIPS2011_1240.pdf)

####Regularization

####Stochastic Nets and GSNs
- [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432)
- [Learning Stochastic Feedforward Neural Networks](http://papers.nips.cc/paper/5026-learning-stochastic-feedforward-neural-networks)
- [Generalized Denoising Auto-Encoders as Generative Models](https://arxiv.org/abs/1305.6663)
- [Deep Generative Stochastic Networks Trainable by Backprop ](https://arxiv.org/abs/1306.1091)

####Others
- [Slow, Decorrelated Features for Pretraining Complex Cell-like Networks](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2009_0933.pdf)
- [What Regularized Auto-Encoders Learn from the Data Generating Distribution ](https://arxiv.org/abs/1211.4246)
- [Generalized Denoising Auto-Encoders as Generative Models ](https://arxiv.org/abs/1305.6663)
- [Why the logistic function?](http://www.cs.berkeley.edu/~jordan/papers/uai.ps)

####Recurrent Nets
- [Learning long-term dependencies with gradient descent is difficult](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf)
- [Advances in Optimizing Recurrent Networks ](https://arxiv.org/abs/1212.0901)
- [Learning recurrent neural networks with Hessian-free optimization](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Martens_532.pdf)
- [On the importance of momentum and initialization in deep learning](http://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf)
- [Long short-term memory (Hochreiter & Schmidhuber)](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Generating Sequences With Recurrent Neural Networks](http://arxiv.org/abs/1308.0850)
- [Long Short-Term Memory in Echo State Networks: Details of a Simulation Study](http://minds.jacobs-university.de/sites/default/files/uploads/papers/2478_Jaeger12.pdf)
- [The "echo state" approach to analysing and training recurrent neural networks](http://minds.jacobs-university.de/sites/default/files/uploads/papers/EchoStatesTechRep.pdf)
- [Backpropagation-Decorrelation: online recurrent learning with O(N) complexity](http://ni.www.techfak.uni-bielefeld.de/files/Steil2004-BDO.pdf)
- [New results on recurrent network training:Unifying the algorithms and accelerating convergence](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0CDsQFjAB&url=http%3A%2F%2Fwww.researchgate.net%2Fpublication%2F2375393_New_Results_on_Recurrent_Network_Training_Unifying_the_Algorithms_and_Accelerating_Convergence%2Ffile%2F9fcfd50fed618a36b7.pdf&ei=hBHBUs_VNur4yQHiqYHACw&usg=AFQjCNFEP3Y5-E5iygiKMfSO4EnltItX0A&sig2=7r8_uIznXhPKXVfv9yVTPg&bvm=bv.58187178,d.aWc&cad=rja)
- [Audio Chord Recognition with Recurrent Neural Networks](http://www-etud.iro.umontreal.ca/~boulanni/ISMIR2013.pdf)
- [Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription]((http://arxiv.org/pdf/1206.6392))

####Convolutional Nets
- [http://www.iro.umontreal.ca/~bengioy/DLbook/convnets.html](http://www.iro.umontreal.ca/~bengioy/DLbook/convnets.html)
- [Generalization and Network Design Strategies (LeCun)](http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf)
- [ImageNet Classification with Deep Convolutional Neural Networks, Alex Krizhevsky, Ilya ](http://books.nips.cc/papers/files/nips25/NIPS2012_0534.pdf) Sutskever, Geoffrey E Hinton, NIPS 2012.
- [On Random Weights and Unsupervised Feature Learning](http://www.stanford.edu/~asaxe/papers/Saxe et al. - 2010 - On Random Weights and Unsupervised Feature Learning.pdf)

####Optimization issues with DL
- [Curriculum Learning](http://www.machinelearning.org/archive/icml2009/papers/119.pdf)
- [Evolving Culture vs Local Minima](http://arxiv.org/pdf/1203.2990v2.pdf)
- [Knowledge Matters: Importance of Prior Information for Optimization](http://arxiv.org/pdf/1301.4083v6.pdf)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
- [Practical recommendations for gradient-based training of deep architectures ](https://arxiv.org/abs/1206.5533)
- [Natural Gradient Works Efficiently (Amari 1998)](http://www.maths.tcd.ie/~mnl/store/Amari1998a.pdf)
- Hessian Free
- Natural Gradient (TONGA)
- [Revisiting Natural Gradient](http://arxiv.org/abs/1301.3584)

####NLP + DL
- [Natural Language Processing (Almost) from Scratch](http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/35671.pdf)
- [DeViSE: A Deep Visual-Semantic Embedding Model](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model)
- [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)
- [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2011_0538.pdf)

####CV+RBM
- [Fields of Experts](http://www.gris.informatik.tu-darmstadt.de/~sroth/pubs/foe-ijcv.pdf)
- [What makes a good model of natural images?](http://people.csail.mit.edu/billf/papers/foe-final.pdf)
- [Phone Recognition with the mean-covariance restricted Boltzmann machine](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2010_0160.pdf)
- [Unsupervised Models of Images by Spike-and-Slab RBMs](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Courville_591.pdf)

<a namDLe="CV + DL"/>

####CV + DL
- [Imagenet classiﬁcation with deep convolutional neural networks](http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf)
- [Learning to relate images](http://www.iro.umontreal.ca/~memisevr/pubs/pami_relational.pdf)

####Scaling Up
- [Large Scale Distributed Deep Networks](http://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)
- [Random search for hyper-parameter optimization](http://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
- [Practical Bayesian Optimization of Machine Learning Algorithms](http://www.cs.toronto.edu/~jasper/bayesopt.pdf)

####DL + Reinforcement learning
- [Playing Atari with Deep Reinforcement Learning (paper not officially released yet!)](http://arxiv.org/abs/1312.5602)


####Graphical Models Background
- [An Introduction to Graphical Models (Mike Jordan, brief course notes)](http://www.cis.upenn.edu/~mkearns/papers/barbados/jordan-tut.pdf)
- [A View of the EM Algorithm that Justifies Incremental, Sparse and Other Variants (Neal &Hinton, important paper to the modern understanding of Expectation-Maximization)](http://www.cs.toronto.edu/~radford/ftp/emk.pdf) 
- [A Unifying Review of Linear Gaussian Models (Roweis & Ghahramani, ties together PCA, factor analysis, hidden Markov models, Gaussian mixtures, k-means, linear dynamical systems](http://authors.library.caltech.edu/13697/1/ROWnc99.pdf)
- [An Introduction to Variational Methods for Graphical Models (Jordan et al, mean-field, etc.)](http://www.cs.berkeley.edu/~jordan/papers/variational-intro.pdf)

####Writing
- [Writing a great research paper (video of the presentation)](https://www.youtube.com/watch?v=g3dkRsTqdDA)

####Software documentation
- [Python](http://www.deeplearning.net/software/theano/tutorial/python.html), [Theano](http://www.deeplearning.net/software/theano/tutorial/), [Pylearn2](http://www.deeplearning.net/software/pylearn2/#documentation)
- [Linux (bash)](http://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO.html)(at least the 5 first sections), [git](http://git-scm.com/book) (5 first sections), [github/contributing to it](http://deeplearning.net/software/theano/dev_start_guide.html#dev-start-guide) (Theano doc), [vim tutorial](http://blog.interlinked.org/tutorials/vim_tutorial.html) or [emacs tutorial](http://www2.lib.uchicago.edu/keith/tcl-course/emacs-tutorial.html)

####Software lists of built-in commands/functions
- [Bash commands](http://ss64.com/bash/)
- [List of Built-in Python Functions](http://docs.python.org/2/library/functions.html)
- [vim commands](http://tnerual.eriogerg.free.fr/vimqrc.html)

####Other Software stuff to know about:
- screen/tmux
- ssh
- ipython
- matplotlib
