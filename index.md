<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Stargazers][stars-shield]][stars-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->

# 机器学习

## 基本概念

机器学习是人工智能领域的一种方法，它利用算法和统计模型来使计算机系统根据输入数据进行预测或做出决策，而无需使用明确的指令。它主要依赖于模式识别和计算学习理论。

机器学习的核心思想是从数据中学习并对新的数据做出决策或预测。

### 它的过程通常包含以下几个步骤
+ 数据收集：收集需要用于训练模型的数据。数据可以是标注的（用于监督学习）或非标注的（用于非监督学习）。
+ 数据预处理：清洗数据，去除噪音和异常值，选择合适的特征，并将数据格式化以适应特定算法的要求。
+ 选择模型：选择合适的机器学习模型。常见的算法包括决策树、神经网络、支持向量机（SVM）、随机森林等。
+ 训练模型：使用训练数据不断调整模型的权重和参数，直到模型能够在训练数据上获得较好的预测性能。
+ 评估模型：使用验证集或测试集来评估模型的性能，确保模型能够在之前未见过的数据上获得良好表现。
+ 参数调整与优化：如果模型表现不佳，可以通过调整参数或选择不同的算法来优化模型。
+ 部署模型：将训练好的模型应用于实际问题中，进行预测或决策。

### 机器学习可以分为几种类型
+ 监督学习（Supervised Learning）：数据集是由输入和输出对组成的，模型的任务是学习输入和输出之间的映射，进行预测。
+ 无监督学习（Unsupervised Learning）：数据集只包含输入，没有标签输出；模型需要自行找出数据的结构，常见的任务有聚类和密度估计。
+ 半监督学习（Semi-Supervised Learning）：结合了监督学习和无监督学习的特性，其中只有一部分数据被标记。
+ 增强学习（Reinforcement Learning）：模型需要通过与环境的交互，根据奖励信号来学习最优策略。


-------------------

## 数学基础

### 参考文档

[高等数学](机器学习/markdown/高等数学.md)

[线性代数](机器学习/markdown/线性代数.md)

[概率论](机器学习/markdown/概率论.md)

### 基本介绍
机器学习中的数学基础涵盖了多个数学领域，包括微积分、线性代数、概率论、统计学和优化理论。

#### 微积分
微积分（Calculus），数学概念，是高等数学中研究函数的微分（Differentiation）、积分（Integration）以及有关概念和应用的数学分支。
它是数学的一个基础学科，内容主要包括极限、微分学、积分学及其应用。
微分学包括求导数的运算，是一套关于变化率的理论。它使得函数、速度、加速度和曲线的斜率等均可用一套通用的符号进行讨论。
积分学，包括求积分的运算，为定义和计算面积、体积等提供一套通用的方法。

#### 线性代数
线性代数是代数学的一个分支，主要处理线性关系问题。线性关系意即数学对象之间的关系是以一次形式来表达的。
例如，在解析几何里，平面上直线的方程是二元一次方程； 空间平面的方程是三元一次方程，而空间直线视为两个平面相交，由两个三元一次方程所组成的方程组来表示。
含有n个未知量的一次方程称为线性方程。关于变量是一次的函数称为线性函数。线性关系问题简称线性问题。解线性方程组的问题是最简单的线性问题。

#### 概率与统计
事件的概率是衡量该事件发生的可能性的量度。虽然在一次随机试验中某个事件的发生是带有偶然性的，但那些可在相同条件下大量重复的随机试验却往往呈现出明显的数量规律。

统计学是通过搜索、整理、分析、描述数据等手段，以达到推断所测对象的本质，甚至预测对象未来的一门综合性科学。
统计学用到了大量的数学及其它学科的专业知识，其应用范围几乎覆盖了社会科学和自然科学的各个领域。

----------

## 理论知识

### 参考文档

[目录](机器学习/markdown/SUMMARY.md)

[单变量线性回归](机器学习/markdown/week1)

[多变量线性回归](机器学习/markdown/week2)

[逻辑回归与正则化](机器学习/markdown/week3)

[神经网络(深度学习)](机器学习/markdown/week4)

[神经网络的学习](机器学习/markdown/week5)

[误差分析](机器学习/markdown/week6)

[支持向量机](机器学习/markdown/week7)

[聚类与降维](机器学习/markdown/week8)

[异常检测](机器学习/markdown/week9)

[应用实例](机器学习/markdown/week10)

[完整笔记](机器学习/Direction.md)

### 基本介绍
机器学习是人工智能的一个重要分支，研究如何通过数据和算法使计算机系统具备学习和自动改进的能力。它涉及从数据中提取模式、构建预测模型和做出决策的技术和方法。

+ 监督学习 (Supervised Learning)： 
    + 这是一种常见的机器学习方法，情况下，训练数据包含输入以及对应的期望输出，目标是学习一个将输入映射到输出的模型。例如分类任务，如垃圾邮件识别，或回归任务，比如房价预测。
+ 无监督学习 (Unsupervised Learning)： 
    + 与监督学习不同，这种方式的训练数据没有任何标签。这类算法的目的通常是挖掘数据中的模式或结构，例如聚类和降维。
+ 半监督学习 (Semi-supervised Learning)： 
    + 这类学习方法适用于大部分数据是无标签的，而小部分数据是有标签的情况。其目标是利用大量无标签数据来帮助生成有意义的模型。
+ 强化学习 (Reinforcement Learning)： 
    + 这是一种机器学习方法，模型通过与环境交互进行学习。每进行一个操作，都会得到一个奖赏或者惩罚。模型的目标是最大化累计奖励。
+ 深度学习 (Deep Learning)： 
    + 深度学习是一类特殊的机器学习算法，它建立了起源于人脑神经网络的模型。这些模型具有众多的层，可以对高维数据进行高效处理。
+ 迁移学习 (Transfer Learning)： 
    + 在这种情况下，机器使用预先训练过的模型作为初始点，然后对新的相关任务进行学习。这有助于减少需要大量标记数据的需要。
+ 异常检测 (Anomaly Detection)： 
    + 这种方法用来确定数据中异常的、罕见的或可疑的模式。例如，识别信用卡欺诈或者机器故障。
+ 特征工程 (Feature Engineering)： 
    + 这是一个过程，它涉及创建和使用一组有用的特征（属性、指标或标量），来有效地描述数据和解决问题。

----------

## 编程与案例实现

### 参考文档

[NumPy](http://c.biancheng.net/numpy/)

[Pandas 数据分析](http://c.biancheng.net/pandas/)

[Jupyter仓库地址](https://github.com/jupyter)

[Jupyter中文文档](https://www.osgeo.cn/jupyter/user-documentation.html)

[SkLearn](https://scikit-learn.org/stable/index.html)

[Python 案例](机器学习/Python实现/readme.md)

<!-- [Python + Jupyter 案例实现](机器学习/Python实现/readme.md) -->

### 基本介绍

#### Python
Python 是一个高层次的结合了解释性、编译性、互动性和面向对象的脚本语言。
Python 的设计具有很强的可读性，相比其他语言经常使用英文关键字，其他语言的一些标点符号，它具有比其他语言更有特色语法结构。

#### NumPy
NumPy（Numerical Python）是Python科学计算的核心库之一，提供了高性能的多维数组对象和各种数学函数，以及对这些数组进行操作的工具。

#### Pandas
Pandas是一个基于NumPy构建的开源数据分析和数据处理库，为Python提供了高效且灵活的数据结构和数据操作工具。

#### Jupyter
Jupyter是一个基于Web的交互式计算环境，支持多种编程语言，包括Python、R和Julia等。
它提供了一个交互式的界面，使用户可以在浏览器中编写和运行代码，并实时查看结果。

#### SkLearn
Scikit-learn，简称Sklearn，是一个用于机器学习的Python库。
它提供了一系列用于数据预处理、特征选择、模型训练和评估的工具和算法。
Scikit-learn建立在NumPy、SciPy和Matplotlib等科学计算库的基础上，并与其它数据科学工具（如Pandas）无缝集成。

----------

## 深度学习(神经网络)

### 参考文档

[资料索引](深度学习/README.md)

[TensorFlow](https://www.tensorflow.org/guide?hl=zh-cn)

[PyTorch](https://pytorch.apachecn.org/)

### 基本介绍

#### TensorFlow
TensorFlow 是一个广泛使用的开源机器学习框架，由 Google Brain 团队开发和维护。它提供了一种灵活且高效的方式来构建、训练和部署机器学习模型。

#### PyTorch
PyTorch 是一个开源的深度学习框架，由 Facebook 的人工智能研究团队开发和维护。它提供了一个灵活且易于使用的平台，用于构建、训练和部署深度神经网络模型。

----------

## 数据挖掘

### 参考文档

[资料总览](数据挖掘/Direction.md)

### 基本介绍
数据挖掘是从大量数据中发现模式、关联、规律和知识的过程。 它结合了统计学、机器学习、人工智能和数据库技术等多个领域的方法和技术，旨在从数据中提取有价值的信息和洞察。

### 数据挖掘的一般流程
+ 问题定义 
    + 首先，确定需要解决的具体问题或目标。这可能涉及到预测、分类、聚类、关联规则挖掘等不同类型的任务。
+ 数据收集 
    + 收集与问题相关的数据，并进行清理和预处理。这包括去除噪声、处理缺失值、解决数据不一致性等步骤。
+ 数据理解 
    + 对数据进行探索性分析，了解数据的特征、属性之间的关系，以及可能存在的模式和异常。
+ 特征选择与转换 
    + 根据问题的要求，选择最相关的特征，并进行必要的转换和归一化操作，以便更好地表示数据。
+ 模型构建与训练 
    + 根据问题类型选择适当的算法和模型，例如决策树、支持向量机、神经网络等。使用训练数据对模型进行训练，并调整参数以优化性能。
+ 模型评估与验证 
    + 使用测试数据集对训练好的模型进行评估，检查其在新数据上的表现。评估指标可以根据问题类型而异，如准确率、召回率、F1 分数等。
+ 模型部署与应用 
    + 将训练好的模型应用于实际场景中，以解决实际问题。这可能涉及将模型集成到软件系统中、构建预测模型或生成报告等。

----------

## Stable Diffusion

### 参考文档

[WebUI 已集成Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

[Diffusion 模块网站](https://civitai.com/)

### 基本介绍
Stable Diffusion是2022年发布的深度学习文本到图像生成模型。
它主要用于根据文本的描述产生详细图像，尽管它也可以应用于其他任务，如内补绘制、外补绘制，以及在提示词指导下产生图生图的转变。


<!-- links -->
[your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/worst001/mkdocs_machine_learning.svg?style=flat-square
[contributors-url]: https://github.com/worst001/mkdocs_machine_learning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/worst001/mkdocs_machine_learning.svg?style=flat-square
[forks-url]: https://github.com/worst001/mkdocs_machine_learning/network/members
[stars-shield]: https://img.shields.io/github/stars/worst001/mkdocs_machine_learning.svg?style=social
[stars-url]: https://github.com/worst001/mkdocs_machine_learning/stargazers
[issues-shield]: https://img.shields.io/github/issues/worst001/mkdocs_machine_learning.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/worst001/mkdocs_machine_learning.svg
[license-shield]: https://img.shields.io/github/license/worst001/mkdocs_machine_learning.svg?style=flat-square
[license-url]: https://github.com/worst001/mkdocs_machine_learning/blob/main/LICENSE.txt
