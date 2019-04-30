---
layout: post
title: "贝叶斯分类器介绍"
categories:
  - Machine Learning
tags:
  - Supervised Learning
date: 2019-04-29
---

### 原理

假设每组数据$$(\vec{x_{(k)}},y_k),k=1,2,\cdots,n$$是从分布$$D$$中独立抽取的（分布$$D$$未知）

对任一分类器$$f$$，它的预测误差可表示为$$P(f(\vec{x})\neq{y})=E[I(f(\vec{x})\neq{y})]=E[E[I(f(\vec{x})\neq{y})\lvert{\vec{x}}]]$$（注：$$(\vec{x},y)\sim{D}$$）

对任一固定的值$$\vec{x_*}$$，$$E[I(f(\vec{x})\neq{y})\lvert{\vec{x}=\vec{x_*}}]=\sum_{i=1}^KP(y=i\lvert{\vec{x}=\vec{x_*}})I(f(\vec{x_*})\neq{i})$$，其中$$K$$为分类个数

使得$$E[I(f(\vec{x})\neq{y})\lvert{\vec{x}=\vec{x_*}}]$$最小的分类器为$$\hat{f}(\vec{x_*})=argmax_{i\in{1,2,\cdots,K}}P(y=i\lvert{\vec{x}=\vec{x_*}})$$。对所有$$\vec{x}$$的取值均满足这一性质的分类器即为贝叶斯分类器，它在所有分类器中使得预测误差$$P(f(\vec{x})\neq{y})$$最小

由贝叶斯法则可以推导出贝叶斯分类器$$\hat{f}(\vec{x_*})=argmax_{i\in{1,2,\cdots,K}}\underbrace{P(y=i)}_\text{class prior}\underbrace{P(\vec{x}=\vec{x_*}\lvert{y=i})}_\text{data likelihood}$$

### 朴素贝叶斯

朴素贝叶斯假设数据的特征是条件相互独立的，即$$P(\vec{x}=\vec{x_*}\lvert{y=i})=\prod_{j=1}^dP(x_j=x_{*j}\lvert{y=i})$$，其中$$d$$为特征个数

以垃圾邮件的分类问题为例，每个特征代表对应的单词在邮件中的出现次数，$$y=0$$表示正常邮件，$$y=1$$表示垃圾邮件

![img](/img/spam.PNG)

假设$$P(\vec{x}=\vec{x_*}\lvert{y=i})=\prod_{j=1}^dP(x_j=x_{*j}\lvert{y=i})=\prod_{j=1}^dPoisson(x_{*j}\lvert{\lambda_j^{(i)}})$$，并定义$$\pi^{(i)}=P(y=i)$$，则问题变为对参数$$\pi^{(i)}$$和$$\lambda_j^{(i)}, i\in\{0,1\},j\in\{1,2,\cdots,d\}$$的求解

使用最大似然估计对参数进行求解，即$$\begin{align*}
argmax_{\pi^{(i)},\lambda_j^{(i)}}\sum_{k=1}^nlnP(\vec{x_{(k)}},y_k) &=argmax_{\pi^{(i)},\lambda_j^{(i)}}\sum_{k=1}^n[ln\pi^{(y_k)}+\sum_{j=1}^d(x_{kj}ln\lambda_j^{(y_k)}-\lambda_j^{(y_k)})] \\ &=argmax_{\pi^{(i)},\lambda_j^{(i)}}\sum_{i=0}^1\sum_{k\lvert{y_k=i}}[ln\pi^{(i)}+\sum_{j=1}^d(x_{kj}ln\lambda_j^{(i)}-\lambda_j^{(i)})]\end{align*}
$$

令$$\pi^{(1)}=1-\pi^{(0)}$$，并令上式的一阶导数为0，可以得到参数的估计值为$$\hat{\pi}^{(i)}=\frac{\sum_{k=1}^nI(y_k=i)}{n}$$，$$\hat{\lambda}_j^{(i)}=\frac{\sum_{k\lvert{y_k=i}}x_{kj}}{\sum_{k=1}^nI(y_k=i)}$$

综上所述求得的朴素贝叶斯分类器为$$\hat{f}(\vec{x_*})=argmax_{i\in{0,1}}[\hat{\pi}^{(i)}\prod_{j=1}^dPoisson(x_{*j}\lvert{\hat{\lambda}_j^{(i)}})]$$

### LDA和QDA

线性判别分析LDA和二次判别分析QDA都假设每个类内的特征符合多维高斯分布，即$$P(\vec{x}\lvert{y=i})=\frac{1}{(2\pi)^{d/2}\lvert{C_i}\rvert^{1/2}}e^{-\frac{1}{2}(\vec{x}-\vec{\mu}_i)^TC_i^{-1}(\vec{x}-\vec{\mu}_i)}$$

LDA与QDA的区别是LDA假设每个类内的协方差矩阵相等，即$$C_1=C_2=\cdots=C_K=C$$，LDA还可以用于高维数据的降维，具体介绍可参考文章[PCA与LDA介绍](https://www.cnblogs.com/sunwq06/p/10787846.html)

具体的推导和计算过程与朴素贝叶斯并没有本质区别，这里就不再详述了，感兴趣的可以参考[The Elements of Statistical Learning(2nd Edition)](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf) P106-P111
