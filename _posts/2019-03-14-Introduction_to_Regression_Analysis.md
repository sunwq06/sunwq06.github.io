---
layout: post
title: "回归分析介绍"
categories:
  - Machine Learning
  - Statistics
tags:
  - Supervised Learning
date: 9999-03-14
---

### 一、线性回归基础

1. 模型介绍与求解

   + 线性回归模型可写为$$y=\vec{x}\cdot\vec{\beta}+\epsilon$$，其中$$\epsilon\sim{N(0,\sigma^2)}$$，$$\vec{\beta}=\begin{bmatrix}\begin{smallmatrix}\beta_0 \\ \beta_1 \\ \vdots \\ \beta_k\end{smallmatrix}\end{bmatrix}$$为待求系数。

   + 假定训练数据集$$X=\begin{bmatrix}\begin{smallmatrix}\vec{x_1}\\\vec{x_2}\\\vdots\\\vec{x_n}\end{smallmatrix}\end{bmatrix}=\begin{bmatrix}\begin{smallmatrix} 1&x_{11}&x_{12}&\cdots&x_{1k} \\ 1&x_{21}&x_{22}&\cdots&x_{2k} \\ \vdots&\vdots&\vdots&\cdots&\vdots \\ 1&x_{n1}&x_{n2}&\cdots&x_{nk} \end{smallmatrix}\end{bmatrix}$$，则有$$\vec{y}=\begin{bmatrix}\begin{smallmatrix}y_0 \\ y_1 \\ \vdots \\ y_n\end{smallmatrix}\end{bmatrix}=X\cdot\vec{\beta}+\begin{bmatrix}\begin{smallmatrix}\epsilon_0 \\ \epsilon_1 \\ \vdots \\ \epsilon_n\end{smallmatrix}\end{bmatrix}$$，其中$$\epsilon_i$$相互独立并且$$\epsilon_i\sim{N(0,\sigma^2)}$$

   + 可以使用最小二乘法求解$$\vec{\beta}$$，即$$argmin_{\beta_0,\beta_1,...,\beta_k}[(\vec{y}-X\vec{\beta})^T(\vec{y}-X\vec{\beta})]$$，求导可得$$X^T(\vec{y}-X\vec{\beta})=0$$，即$$\vec{\beta}$$的估计值$$\hat{\vec{\beta}}=(X^TX)^{-1}X^T\vec{y}$$

2. 回归系数的统计解释

   + $$\sigma^2$$的估计值$$s^2=\frac{1}{n-k-1}\sum_{i=1}^ne_i^2=\frac{1}{n-k-1}\sum_{i=1}^n(y_i-\hat{y}_i)^2$$，其中$$\hat{y}_i$$为预测值，$$e_i$$为留数(即residual)，并且$$\frac{(n-k-1)s^2}{\sigma^2}\sim{\chi^2_{n-k-1}}$$

   + $$\hat{\vec{\beta}}$$的协方差矩阵$$Cov(\hat{\vec{\beta}})=[(X^TX)^{-1}X^T]Cov(\vec{y})[(X^TX)^{-1}X^T]^T=\sigma^2(X^TX)^{-1}$$，协方差矩阵的估计值$$\hat{Cov}(\hat{\vec{\beta}})=s^2(X^TX)^{-1}$$

   + 对一个新数据$$\vec{x}_*$$，有预测值$$y_*=\vec{x}_*\cdot\vec{\beta}+\epsilon_*=E(y_*)+\epsilon_*$$，$$y_*$$的估计值$$\hat{y}_*=\vec{x}_*\cdot\hat{\vec{\beta}}$$，预测值$$y_*$$的置信水平为$$1-\alpha$$的置信区间为$$[\text{    }\hat{y}_*-t_{n-k-1,\alpha/2}\sqrt{\hat{Var}(\hat{y}_*-y_*)}\text{,       }\hat{y}_*+t_{n-k-1,\alpha/2}\sqrt{\hat{Var}(\hat{y}_*-y_*)}\text{    }]$$，其中$$\hat{Var}(\hat{y}_*-y_*)=\vec{x}_*\hat{Cov}(\hat{\vec{\beta}}){\vec{x}_*}^T+s^2=s^2[\vec{x}_*(X^TX)^{-1}{\vec{x}_*}^T+1]$$

   + 单个系数的假设检验$$H_0\text{: }\beta_j=d\text{   vs   }H_1\text{: }\beta_j\neq{d}$$，检验统计量$$t(\hat{\beta}_j)=\frac{\hat{\beta}_j-d}{\sqrt{[\hat{Cov}(\hat{\vec{\beta}})]_{j+1,j+1}}}$$在假设$$H_0$$下满足分布$$t_{n-k-1}$$，其中$$[\hat{Cov}(\hat{\vec{\beta}})]_{j+1,j+1}$$表示$$\hat{Cov}(\hat{\vec{\beta}})$$的第$$j+1$$个对角元素。若$$\lvert{t(\hat{\beta}_j)\rvert>t_{n-k-1,\alpha/2}}$$，则在显著性水平为$$\alpha$$时拒绝原假设$$H_0$$

   + 多个系数的假设检验(Generalized F-test)

     不失一般性，令$$H_0\text{: }最后q个系数均为0(即\beta_{k-q+1}=\beta_{k-q+2}=\cdots=\beta_{k}=0)\text{  vs  }H_1\text{: }最后q个系数中至少有一个不为0$$

     定义$$RSS=\sum_{i=1}^ne_i^2=\sum_{i=1}^n(y_i-\hat{y}_i)^2$$，记$$RSS_0$$为使用简化后的模型（$$\beta_{k-q+1}=\beta_{k-q+2}=\cdots=\beta_{k}=0$$）计算出的RSS，$$RSS_1$$为使用原始模型计算出的RSS，则检验统计量$$F=\frac{(RSS_0-RSS_1)/q}{RSS_1/(n-k-1)}$$在假设$$H_0$$下满足分布$$F_{q,n-k-1}$$，若$$F>F_{q,n-k-1,\alpha}$$，则在显著性水平为$$\alpha$$时拒绝原假设$$H_0$$

### 二、线性回归的模型诊断
