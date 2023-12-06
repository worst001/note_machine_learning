# 机器学习

整理了机器学习相关资料与手册，包括数学基础、机器学习模型实现示例、神经网络。

公网资料、笔记地址请访问这里 


<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->

--------------------

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

--------------------

## 目录

[目录与大纲](index.md)

## 数学基础

+ [高等数学](机器学习/markdown/高等数学.md)
+ [线性代数](机器学习/markdown/线性代数.md)
+ [概率论](机器学习/markdown/概率论.md)


## 理论知识

+ [目录](机器学习/markdown/SUMMARY.md)
+ [单变量线性回归](机器学习/markdown/week1.md)
+ [多变量线性回归](机器学习/markdown/week2.md)
+ [逻辑回归与正则化](机器学习/markdown/week3.md)
+ [神经网络(深度学习)](机器学习/markdown/week4.md)
+ [神经网络的学习](机器学习/markdown/week5.md)
+ [误差分析](机器学习/markdown/week6.md)
+ [支持向量机](机器学习/markdown/week7.md)
+ [聚类与降维](机器学习/markdown/week8.md)
+ [异常检测](机器学习/markdown/week9.md)
+ [应用实例](机器学习/markdown/week10.md)
+ [完整笔记](机器学习/机器学习个人笔记完整版v5.52.pdf)


## 编程与案例实现

+ [NumPy](http://c.biancheng.net/numpy/)
+ [Pandas 数据分析](http://c.biancheng.net/pandas/)
+ [Jupyter仓库地址](https://github.com/jupyter)
+ [Jupyter中文文档](https://www.osgeo.cn/jupyter/user-documentation.html)
+ [SkLearn](https://scikit-learn.org/stable/index.html)
+ [Python 案例](机器学习/Python实现/readme.md)

<!-- [Python + Jupyter 案例实现](机器学习/Python实现/readme.md) -->

## 深度学习(神经网络)

+ [资料总览](深度学习/README.md)
+ [TensorFlow](https://www.tensorflow.org/guide?hl=zh-cn)
+ [PyTorch](https://pytorch.apachecn.org/)


## 数据挖掘

+ [资料总览](数据挖掘/Python数据分析与挖掘实战.pdf)


## Stable Diffusion

+ [WebUI 已集成Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
+ [Diffusion 模块网站](https://civitai.com/)

-------------------

## 贡献者

请阅读**CONTRIBUTING.md** 查阅为该项目做出贡献的开发者。

#### 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

<!-- ## 作者 -->
<!--  -->
<!-- [小昊子](https://github.com/worst001) -->
<!--  -->
<!-- 制做不易，如果有帮到你就请作者喝杯咖啡吧! -->
<!--  -->
<!-- ![支付宝加微信](https://xiyou-oss.oss-cn-shanghai.aliyuncs.com/%E5%85%AC%E4%BC%97%E5%8F%B7%E4%B8%8E%E6%94%AF%E4%BB%98/%E6%94%AF%E4%BB%98%E5%AE%9D%E5%8A%A0%E5%BE%AE%E4%BF%A1.jpg) -->
<!--  -->
<!-- 作者无聊时做的测试游戏，完全免费哦！ -->
<!--  -->
<!-- ![公众号](https://xiyou-oss.oss-cn-shanghai.aliyuncs.com/%E5%85%AC%E4%BC%97%E5%8F%B7%E4%B8%8E%E6%94%AF%E4%BB%98/%E5%85%AC%E4%BC%97%E5%8F%B7%E5%B0%8F.jpg) -->

## 参考资料

[https://github.com/lawlite19/MachineLearning_Python](https://github.com/lawlite19/MachineLearning_Python)

[https://github.com/MorvanZhou/tutorials](https://github.com/MorvanZhou/tutorials)

[https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes)

<!-- links -->
[your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/worst001/mkdocs_machine_learning.svg?style=flat-square
[contributors-url]: https://github.com/worst001/mkdocs_machine_learning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/worst001/mkdocs_machine_learning.svg?style=flat-square
[forks-url]: https://github.com/worst001/mkdocs_machine_learning/network/members
[stars-shield]: https://img.shields.io/github/stars/worst001/mkdocs_machine_learning.svg?style=flat-square
[stars-url]: https://github.com/worst001/mkdocs_machine_learning/stargazers
[issues-shield]: https://img.shields.io/github/issues/worst001/mkdocs_machine_learning.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/worst001/mkdocs_machine_learning.svg
[license-shield]: https://img.shields.io/github/license/worst001/mkdocs_machine_learning.svg?style=flat-square
[license-url]: https://github.com/worst001/mkdocs_machine_learning/blob/main/LICENSE.txt
<!-- [linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555 -->
<!-- [linkedin-url]: https://linkedin.com/in/shaojintian -->
