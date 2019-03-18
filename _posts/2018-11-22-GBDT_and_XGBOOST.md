---
layout: post
title: "GBDT和XGBOOST算法原理"
tags: [机器学习]
date: 2018-11-22
---

### GBDT

本文以多分类问题为例介绍GBDT的算法，针对多分类问题，每次迭代都需要生成K个树（K为分类的个数），记为$$F_{mk}(x)$$，其中m为迭代次数，k为分类。

针对每个训练样本，使用的损失函数通常为$$L(y_i, F_{m1}(x_i), ..., F_{mK}(x_i))=-\sum_{k=1}^{K}I({y_i}=k)ln[p_{mk}(x_i)]=-\sum_{k=1}^{K}I({y_i}=k)ln(\frac{e^{F_{mk}(x_i)}}{\sum_{l=1}^{K}e^{F_{ml}(x_i)}})$$，此时损失函数的梯度可以表示为$$g_{mki}=-\frac{\partial{L(y_i, F_{m1}(x_i), ..., F_{mK}(x_i))}}{\partial{F_{mk}(x_i)}}=I({y_i}=k)-p_{mk}(x_i)=I({y_i}=k)-\frac{e^{F_{mk}(x_i)}}{\sum_{l=1}^{K}e^{F_{ml}(x_i)}}$$。

GBDT算法的流程如下所示：
1. for k=1 to K: Initialize $$F_{0k}(x)=0$$
2. for m=1 to M:
  + for k=1 to K: compute $$g_{m-1,ki}$$ for each sample $$(x_i, y_i)$$
  + for k=1 to K: build up regression tree $$R_{mkj}$$(j=1 to J<sub>mk</sub> refer to the leaf nodes) from training samples $$(x_i, g_{m-1,ki})_{i=1,...,N}$$
  + for k=1 to K: compute leaf weights $$w_{mkj}$$ for j=1 to J<sub>mk</sub>
  + for k=1 to K: $$F_{mk}(x)=F_{m-1,k}(x)+\eta*\sum_{j=1}^{J_{mk}}w_{mkj}I({x}\in{R_{mkj}})$$, $$\eta$$为学习率

针对$$w_{mkj}$$的计算，有$$w_{mkj(j=1...J_{mk},k=1...K)}=argmin_{w_{kj(j=1...J_{mk},k=1...K)}}\sum_{i=1}^{N}L(y_i, ...,   F_{m-1,k}(x_i)+\sum_{j=1}^{J_{mk}}w_{kj}I({x_i}\in{R_{mkj}}), ...)$$

为了求得w的值，使上述公式的一阶导数为0，利用Newton-Raphson公式（在这个问题中将初始值设为0，只进行一步迭代，并且Hessian矩阵只取对角线上的值），记$$L_i=L(y_i, ...,   F_{m-1,k}(x_i)+\sum_{j=1}^{J_{mk}}w_{kj}I({x_i}\in{R_{mkj}}), ...)$$，有$$w_{mkj}=-\frac{\sum_{i=1}^N\partial{L_i}/\partial{w_{kj}}}{\sum_{i=1}^N\partial^2{L_i}/\partial{w_{kj}^2}}=\frac{\sum_{i=1}^{N}I({x_i}\in{R_{mkj}})[I({y_i}=k)-p_{m-1,k}(x_i)]}{\sum_{i=1}^{N}I^2({x_i}\in{R_{mkj}})p_{m-1,k}(x_i)[1-p_{m-1,k}(x_i)]}=\frac{\sum_{x_i\in{R_{mkj}}}g_{m-1,ki}}{\sum_{x_i\in{R_{mkj}}}\lvert{g_{m-1,ki}}\rvert(1-\lvert{g_{m-1,ki}}\rvert)}$$

文献[1]在$$w_{mkj}$$的前面乘以了$$\frac{K-1}{K}$$的系数，可能原因是在分类器的建立过程中，可以把任意一个$$F_k(x)$$始终设为0，只计算剩余的$$F_k(x)$$，因此$$w_{mkj} \gets \frac{1}{K}*0+\frac{K-1}{K}w_{mkj}$$

参考文献

[1] Friedman, Jerome H. Greedy function approximation: A gradient boosting machine. Ann. Statist. 29 (2001), 1189--1232.

### XGBOOST

仍以多分类问题介绍XGBOOST的算法，相对于GBDT的一阶导数，xgboost还引入了二阶导数，并且加入了正则项。

区别于上文的符号，记$$g_{mki}=\frac{\partial{L(y_i, F_{m1}(x_i), ..., F_{mK}(x_i))}}{\partial{F_{mk}(x_i)}}=p_{mk}(x_i)-I({y_i}=k)=\frac{e^{F_{mk}(x_i)}}{\sum_{l=1}^{K}e^{F_{ml}(x_i)}}-I({y_i}=k)$$，并记$$h_{mki}=\frac{\partial^2{L(y_i, F_{m1}(x_i), ..., F_{mK}(x_i))}}{\partial{F_{mk}(x_i)}^2}=p_{mk}(x_i)[1-p_{mk}(x_i)]$$，xgboost的计算流程如下：
1. for k=1 to K: Initialize $$F_{0k}(x)=0$$
2. for m=1 to M:
  + for k=1 to K: compute $$g_{m-1,ki}$$ and $$h_{m-1,ki}$$ for each sample $$(x_i, y_i)$$
  + for k=1 to K: compute $$f_{mk}$$(i.e., $$R_{mkj}$$, j=1 to J<sub>mk</sub> refer to the leaf nodes) to minimize the function $$\sum_{i=1}^{N}[g_{m-1,ki}f_{mk}(x_i)+\frac{1}{2}h_{m-1,ki}f_{mk}(x_i)^2]+\Omega(f_{mk})$$, where $$\Omega$$ is the regularization function
  + for k=1 to K: $$F_{mk}(x)=F_{m-1,k}(x)+\eta*\sum_{j=1}^{J_{mk}}w_{mkj}I({x}\in{R_{mkj}})$$, $$\eta$$为学习率

针对回归树$$f_{mk}(x)=\sum_{j=1}^{J_{mk}}w_{mkj}I({x}\in{R_{mkj}})$$的求解，记$$\Omega(f_{mk})=\gamma J_{mk}+\frac \lambda {2}\sum_{j=1}^{J_{mk}}w_{mkj}^2$$，最小化问题变为$$\sum_{j=1}^{J_{mk}}[(\sum_{x_i \in R_{mkj}}g_{m-1,ki})w_{mkj}+\frac{1}{2}(\lambda+\sum_{x_i \in R_{mkj}}h_{m-1,ki})w_{mkj}^2]+\gamma J_{mk}$$，这是J<sub>mk</sub>个独立的二次函数之和。

假设回归树的结构已经固定，记$$T_k=J_{mk},G_{kj}=\sum_{x_i \in R_{mkj}}g_{m-1,ki}和H_{kj}=\sum_{x_i \in R_{mkj}}h_{m-1,ki}$$，此时在$$w_{mkj}=-\frac{G_{kj}}{H_{kj}+\lambda}$$时目标函数取得最小值，最小值为$$-\frac{1}{2}\sum_{j=1}^{T_k}\frac{G_{kj}^2}{H_{kj}+\lambda}+\gamma T_k$$。

使用贪心算法确定回归树的结构，从一个节点开始不断进行分裂，分裂的规则是挑选使分裂增益最大的特征和分裂点进行分裂，直到达到某个停止条件为止。假设待分裂的节点为$$I$$，分裂后的两个节点分别为$$I_L$$和$$I_R$$，那么分裂增益可以表示为：$$-\frac{1}{2}\frac{(\sum_{x_i \in I}g_{m-1,ki})^2}{\lambda+\sum_{x_i \in I}h_{m-1,ki}}+\frac{1}{2}[\frac{(\sum_{x_i \in I_L}g_{m-1,ki})^2}{\lambda+\sum_{x_i \in I_L}h_{m-1,ki}}+\frac{(\sum_{x_i \in I_R}g_{m-1,ki})^2}{\lambda+\sum_{x_i \in I_R}h_{m-1,ki}}]-\gamma$$

XGBOOST的Python包中的重要参数
+ eta(alias:learning_rate): learning rate or shrinkage，同上文的$$\eta$$，常用值0.01~0.2
+ gamma(alias: min_split_loss): min loss reduction to create new tree split，同上文的$$\gamma$$
+ lambda(alias: reg_lambda): L2 regularization on leaf weights，同上文的$$\lambda$$
+ alpha(alias: reg_alpha): L1 regularization on leaf weights
+ max_depth: max depth per tree，常用值3~10
+ min_child_weight: minimum sum of instance weight(hessian) of all observations required in a child，即在一个节点中的样本的损失函数二阶导数之和，以多分类为例，$$\sum_{x_i \in I}h_{mki}=\sum_{x_i \in I}p_{mk}(x_i)[1-p_{mk}(x_i)]$$
+ subsample: % samples used per tree，常用值0.5~1
+ colsample_bytree: % features used per tree，常用值0.5~1
+ 针对样本类别不均衡问题，若是二分类问题，可以设置参数scale_pos_weight，通常设为sum(negative instances)/sum(positive instances); 若是多分类问题，可以通过xgboost.DMatrix(...,weight,...)或者在Scikit-Learn API中调用fit(...,sample_weight,...)
+ 特征重要性可以通过该特征被选中作为节点分裂特征的次数来度量
+ xgboost示例可参见文章[Xgboost应用及调参示例](https://sunwenqi10.github.io/blog/2018/09/21/xgboost_application_and_tuning)
