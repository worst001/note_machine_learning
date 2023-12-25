<a name="readme-top"></a>
<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->

<!-- 项目LOGO -->
<br />
<div align="center">
  <a href="http://mkdocs.grft.top">
    <img src="https://xiyou-oss.oss-cn-shanghai.aliyuncs.com/mkdocs/logo.png" alt="Logo" width="480" height="270">
  </a>

  <h3 align="center">机器学习</h3>

  <p align="center">
    <br />
    <a href="http://mkdocs.grft.top/机器学习/"><strong>探索文档 »</strong></a>
    <br />
  </p>
</div>

<!-- 目录 -->
<details>
  <summary>目录</summary>
  <ol>
    <li><a href="#关于项目">关于项目</a></li>
    <li><a href="#什么是机器学习">什么是机器学习</a></li>
    <li><a href="#技术目录">技术目录</a></li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系方式">联系方式</a></li>
    <li><a href="#鸣谢">鸣谢</a></li>
  </ol>
</details>

## 关于项目

整理了机器学习相关资料与手册，包括数学基础、机器学习模型实现示例、神经网络。

公网资料、笔记地址请访问这里 

- 文档地址: [http://mkdocs.grft.top/机器学习/](http://mkdocs.grft.top/机器学习/)

其他相关技术可以访问我的博客，主页地址请访问这里

- 访问入口：[http://mkdocs.grft.top](http://mkdocs.grft.top)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 什么是机器学习

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

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 技术目录

[目录与大纲](index.md)

### 数学基础

+ [高等数学](机器学习/markdown/高等数学.md)
+ [线性代数](机器学习/markdown/线性代数.md)
+ [概率论](机器学习/markdown/概率论.md)


### 理论知识

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


### 编程与案例实现

+ [NumPy](http://c.biancheng.net/numpy/)
+ [Pandas 数据分析](http://c.biancheng.net/pandas/)
+ [Jupyter仓库地址](https://github.com/jupyter)
+ [Jupyter中文文档](https://www.osgeo.cn/jupyter/user-documentation.html)
+ [SkLearn](https://scikit-learn.org/stable/index.html)
+ [Python 案例](机器学习/Python实现/readme.md)

<!-- [Python + Jupyter 案例实现](机器学习/Python实现/readme.md) -->

### 深度学习(神经网络)

+ [资料总览](深度学习/README.md)
+ [TensorFlow](https://www.tensorflow.org/guide?hl=zh-cn)
+ [PyTorch](https://pytorch.apachecn.org/)


### 数据挖掘

+ [资料总览](数据挖掘/Python数据分析与挖掘实战.pdf)


### Stable Diffusion

+ [WebUI 已集成Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
+ [Diffusion 模块网站](https://civitai.com/)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

<!-- 贡献 -->

## 贡献

贡献是使开源社区成为一个如此令人惊叹的地方，以学习、激励和创造。您所做的任何贡献都将非常感谢。

如果您对使这个项目变得更好有建议，请 fork 该仓库并创建 pull request。您也可以打开一个带有“enhancement”标签的问题。不要忘记给这个项目点个星！再次感谢！

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- 许可证 -->
## 许可证

根据 MIT 许可证进行分发。更多信息请参见 [LICENSE.txt](LICENSE)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

<!-- 联系方式 -->
## 联系方式

关注我: [小昊子](https://github.com/worst001)

博客地址: [http://mkdocs.grft.top](http://mkdocs.grft.top)

项目链接: [https://github.com/worst001/mkdocs_machine_learning](https://github.com/worst001/mkdocs_machine_learning)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 鸣谢

因为仓库与文档的数量比较大，有些借鉴资料忘了在`参考文档`部分提及原作者与原仓库，若有疏漏请告诉，我及时补上。

所有引用的原资料都确认是开源认证，若有侵权请告知。

[https://github.com/lawlite19/MachineLearning_Python](https://github.com/lawlite19/MachineLearning_Python)

[https://github.com/MorvanZhou/tutorials](https://github.com/MorvanZhou/tutorials)

[https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes)

[https://openai.com/chatgpt](https://openai.com/chatgpt)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

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
