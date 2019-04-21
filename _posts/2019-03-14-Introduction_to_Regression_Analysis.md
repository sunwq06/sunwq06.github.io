---
layout: post
title: "回归分析介绍"
categories:
  - Machine Learning
  - Statistics
tags:
  - Supervised Learning
date: 2019-03-14
---

### 一、线性回归基础

1. 模型介绍与求解

   + 线性回归模型可写为$$y=\vec{x}\cdot\vec{\beta}+\epsilon$$，其中$$\vec{\beta}=\begin{bmatrix}\begin{smallmatrix}\beta_0 \\ \beta_1 \\ \vdots \\ \beta_k\end{smallmatrix}\end{bmatrix}$$为待求系数，$$\vec{x}=[1,x_1,x_2,\cdots,x_k]$$，$$\epsilon\sim{N(0,\sigma^2)}$$。

   + 假定训练数据集$$X=\begin{bmatrix}\begin{smallmatrix}\vec{x_{(1)}}\\\vec{x_{(2)}}\\\vdots\\\vec{x_{(n)}}\end{smallmatrix}\end{bmatrix}=\begin{bmatrix}\begin{smallmatrix} 1&x_{(1)1}&x_{(1)2}&\cdots&x_{(1)k} \\ 1&x_{(2)1}&x_{(2)2}&\cdots&x_{(2)k} \\ \vdots&\vdots&\vdots&\cdots&\vdots \\ 1&x_{(n)1}&x_{(n)2}&\cdots&x_{(n)k} \end{smallmatrix}\end{bmatrix}$$，则有$$\vec{y}=\begin{bmatrix}\begin{smallmatrix}y_1 \\ y_2 \\ \vdots \\ y_n\end{smallmatrix}\end{bmatrix}=X\cdot\vec{\beta}+\begin{bmatrix}\begin{smallmatrix}\epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n\end{smallmatrix}\end{bmatrix}$$，其中$$\epsilon_i$$相互独立并且$$\epsilon_i\sim{N(0,\sigma^2)}$$

   + 可以使用最小二乘法求解$$\vec{\beta}$$，即$$argmin_{\beta_0,\beta_1,...,\beta_k}[(\vec{y}-X\vec{\beta})^T(\vec{y}-X\vec{\beta})]$$，求导可得$$X^T(\vec{y}-X\vec{\beta})=0$$，即$$\vec{\beta}$$的估计值$$\hat{\vec{\beta}}=(X^TX)^{-1}X^T\vec{y}$$

2. 回归系数的统计解释

   + $$\sigma^2$$的估计值$$s^2=\frac{1}{n-k-1}\sum_{i=1}^ne_i^2=\frac{1}{n-k-1}\sum_{i=1}^n(y_i-\hat{y}_i)^2$$，其中$$\hat{y}_i$$为预测值，$$e_i$$为留数(即residual)，并且$$\frac{(n-k-1)s^2}{\sigma^2}\sim{\chi^2_{n-k-1}}$$

   + $$\hat{\vec{\beta}}$$的协方差矩阵$$Cov(\hat{\vec{\beta}})=[(X^TX)^{-1}X^T]Cov(\vec{y})[(X^TX)^{-1}X^T]^T=\sigma^2(X^TX)^{-1}$$，协方差矩阵的估计值$$\hat{Cov}(\hat{\vec{\beta}})=s^2(X^TX)^{-1}$$

   + 对一个新数据$$\vec{x}_*$$，有预测值$$y_*=\vec{x}_*\cdot\vec{\beta}+\epsilon_*=E(y_*)+\epsilon_*$$，$$y_*$$的估计值$$\hat{y}_*=\vec{x}_*\cdot\hat{\vec{\beta}}$$，预测值$$y_*$$的置信水平为$$1-\alpha$$的置信区间为$$[\text{    }\hat{y}_*-t_{n-k-1,\alpha/2}\sqrt{\hat{Var}(\hat{y}_*-y_*)}\text{,       }\hat{y}_*+t_{n-k-1,\alpha/2}\sqrt{\hat{Var}(\hat{y}_*-y_*)}\text{    }]$$，其中$$\hat{Var}(\hat{y}_*-y_*)=\vec{x}_*\hat{Cov}(\hat{\vec{\beta}}){\vec{x}_*}^T+s^2=s^2[\vec{x}_*(X^TX)^{-1}{\vec{x}_*}^T+1]$$

   + 单个系数的假设检验$$H_0\text{: }\beta_j=d\text{   vs   }H_1\text{: }\beta_j\neq{d}$$，检验统计量$$t(\hat{\beta}_j)=\frac{\hat{\beta}_j-d}{\sqrt{[\hat{Cov}(\hat{\vec{\beta}})]_{j+1,j+1}}}$$在假设$$H_0$$下满足分布$$t_{n-k-1}$$，其中$$[\hat{Cov}(\hat{\vec{\beta}})]_{j+1,j+1}$$表示$$\hat{Cov}(\hat{\vec{\beta}})$$的第$$j+1$$个对角元素。若$$\lvert{t(\hat{\beta}_j)\rvert>t_{n-k-1,\alpha/2}}$$，则在显著性水平为$$\alpha$$时拒绝原假设$$H_0$$

   + 多个系数的假设检验(Generalized F-test)

     不失一般性，令$$H_0\text{: }最后q个系数均为0(即\beta_{k-q+1}=\beta_{k-q+2}=\cdots=\beta_{k}=0)\text{  vs  }H_1\text{: }最后q个系数中至少有一个不为0$$

     定义$$TSS=\sum_{i=1}^n(y_i-\bar{y})^2$$，$$RSS=\sum_{i=1}^ne_i^2=\sum_{i=1}^n(y_i-\hat{y}_i)^2$$以及$$R^2=\frac{TSS-RSS}{TSS}$$。记$$RSS_0$$为使用简化后的模型（$$\beta_{k-q+1}=\beta_{k-q+2}=\cdots=\beta_{k}=0$$）计算出的RSS，$$RSS_1$$为使用原始模型计算出的RSS，则检验统计量$$F=\frac{(RSS_0-RSS_1)/q}{RSS_1/(n-k-1)}$$在假设$$H_0$$下满足分布$$F_{q,n-k-1}$$，若$$F>F_{q,n-k-1,\alpha}$$，则在显著性水平为$$\alpha$$时拒绝原假设$$H_0$$

### 二、线性回归的模型诊断

1. Influential Points

   + Outlier

     Outliers are unusual observations with respect to the values of the response variable.

     记留数$$e_i=y_i-\hat{y}_i$$，则$$Cov({\vec{e}})=Cov(\vec{y}-X\hat{\vec{\beta}})=Cov((I-H)\vec{y})$$，其中$$H=X(X^TX)^{-1}X^T$$。因此$$Cov({\vec{e}})=(I-H)Cov(\vec{y})(I-H)^T=\sigma^2(I-H)(I-H)^T=\sigma^2(I-H)$$, $$\hat{Var}(e_i)=s^2(1-h_{ii})$$，其中$$h_ii$$是矩阵$$H$$的第$$i$$个对角元素

     可通过留数判断是否为异常值：$$\begin{cases}e_i^{\text{st}}=\frac{e_i}{\sqrt{s^2(1-h_{ii})}}\sim{approximate\text{  } N(0,1)} \\ e_i^{\text{stud}}=\frac{e_i}{\sqrt{s_{(i)}^2(1-h_{ii})}}\sim{t_{n-(k+1)}}, \text{ }s_{(i)}^2为去掉第i个点进行回归后得到的s^2\end{cases}$$

   + Leverage

     Leverage is a measure of influence of an observation solely in terms of its explanatory variables.

     第$$i$$个点的Leverage为$$h_{ii}$$，$$\sum_{i=1}^{n}{h_{ii}}=tr(X(X^TX)^{-1}X^T)=tr((X^TX)^{-1}X^TX)=tr(I_{k+1})=k+1\text{  }$$[注：$$tr(AB)=tr(BA)$$]

     经验公式：High Leverage if $$h_{ii}>\frac{3(k+1)}{n}$$

   + Cook Distance

     $$D_i=\frac{\sum_{j=1}^{n}(\hat{y}_j-\hat{y}_j^{(i)})^2}{(k+1)s^2}=\frac{1}{k+1}\underbrace{(e_i^{\text{st}})^2}_\text{Outlier}\underbrace{\frac{h_{ii}}{1-h_{ii}}}_\text{Leverage}$$，其中$$\hat{y}_j^{(i)}$$为去掉第$$i$$个点进行回归后得到的预测值

     经验公式：High Influential Point if Cook Distance $$D_i\gg\frac{1}{n}$$

2. Heteroscedasticity

   + Breusch-Pagan Test

     假设检验$$H_0: Var(y)=\sigma^2\text{  }$$  vs  $$\text{  }H_1: Var(y)=\sigma^2(\vec{x}\cdot\vec{\gamma})=\sigma^2(\gamma_0+\gamma_1x_1+\gamma_2x_2+\cdots+\gamma_kx_k)$$  

     使用$$e_i^2$$对$$\vec{x_{(i)}}$$做回归，则在假设$$H_0$$下有$$F=\frac{(TSS-RSS)/k}{RSS/(n-k-1)}=\frac{R^2/k}{(1-R^2)/(n-k-1)}\sim{F_{k,n-k-1}}$$

   + White Test

     假设检验$$H_0: Var(y)=\sigma^2\text{  }$$  vs  $$\text{  }H_1: Var(y)=\sigma^2(\gamma_0+\gamma_1\hat{y}+\gamma_2\hat{y}^2)$$

     使用$$e_i^2$$对$$(1,\text{ },\hat{y}_i,\text{ },\hat{y}_i^2)$$做回归，则在假设$$H_0$$下有$$F=\frac{R^2/2}{(1-R^2)/(n-3)}\sim{F_{2,n-3}}$$

   + 对y进行变换有时可以缓解Heteroscedasticity，例如$$ln(y+c)$$或$$\sqrt{y+c}$$，$$c\ge{0}$$

3. Collinearity

   + $$VIF_j=\frac{1}{1-R_j^2},\text{ }j=1,2,\cdots,k$$，其中$$R_j^2$$表示特征$$x_j$$对其余特征$$1,x_1,x_2,\cdots,x_{j-1},x_{j+1},\cdots,x_k$$进行回归后得到的$$R^2$$

   + 经验公式：High Collinearity if $$VIF_j>10(i.e.,\text{ }R_j^2>0.9)$$


### 三、线性回归的特征选择

1. LASSO

   $$argmin_{\beta_0,\beta_1,...,\beta_k}[(\vec{y}-X\vec{\beta})^T(\vec{y}-X\vec{\beta})+\lambda\sum_{i=1}^k\lvert{\beta_i}\rvert]$$，$$\lambda\ge{0}$$

   求解LASSO前需将特征进行标准化(Standardization)处理

2. K-fold Cross Validation   

   当$$K=n$$时称为LOOCV，线性回归的LOOCV可以通过公式表示，即$$\frac{1}{n}\sum_{i=1}^n(\frac{y_i-\hat{y}_i}{1-h_{ii}})^2$$

3. Model Summary Statistics

   + Adjusted R<sup>2</sup>

     $$R_a^2=1-\frac{RSS/(n-p)}{TSS/(n-1)}=1-\frac{n-1}{n-p}(1-R^2)$$（注：模型使用了$$p-1$$个特征，即需求解$$p$$个参数，$$R_a^2\le{R^2}\le{1}$$）

   + Mallow C<sub>p</sub> Statistic

     $$C_p=\frac{RSS_p+2ps_{full}^2}{n}$$，其中$$RSS_p$$表示使用$$p-1$$个特征（即需求解$$p$$个参数）进行回归得到的RSS，$$s_{full}^2$$表示使用所有特征进行回归得到的$$s^2$$

   + AIC and BIC

     $$AIC=\frac{RSS_p+2ps_{full}^2}{ns_{full}^2}$$，$$BIC=\frac{RSS_p+ln(n)ps_{full}^2}{ns_{full}^2}$$

4. Automatic Variable Selection

   定义$$t(\hat{\beta}_j)=\frac{\hat{\beta}_j}{\sqrt{[\hat{Cov}(\hat{\vec{\beta}})]_{j+1,j+1}}}$$，具体说明参见“回归系数的统计解释”这一部分

   + Forward Selection
     + 从常数开始每步在前一步的基础上加入一个特征，可从几个方面决定该特征的选取，例如$$\lvert{t(\hat{\beta}_j)}\rvert$$最大，或者加入该特征后RSS最小（即$$R^2$$最大）
     + 最终从模型$$M_0,M_1,\cdots,M_k$$（下标表示使用的特征数量）中选择最优模型，选择可以通过交叉检验（Cross Validation）进行，也可以使用模型统计量（例如$$R_a^2,\text{ }C_p,\text{ }AIC,\text{ }BIC$$）进行选择

   + Backward Selection
     + 从所有特征开始每步在前一步的基础上删除一个特征，可从几个方面决定该特征的选取，例如$$\lvert{t(\hat{\beta}_j)}\rvert$$最小，或者删除该特征后RSS最小（即$$R^2$$最大）
     + 最终从模型$$M_0,M_1,\cdots,M_k$$中选择最优模型

   + Stepwise Selection
     + 从常数开始每步在前一步的基础上加入一个特征，同其它未加入的特征相比，该特征满足在加入模型后得到的$$\lvert{t(\hat{\beta}_j)}\rvert$$最大，并且可以通过$$\beta_j\ne{0}$$的显著性检验，具体说明参见“回归系数的统计解释”这一部分
     + 加入新特征后在新模型中查看是否有之前加入的特征不能通过显著性检验，如果有就从模型中删除这些特征
     + 重复上述两个步骤直到不能再添加或删除任一特征为止，将每步所得到的模型进行比较，从中选择最优模型

### 四、广义线性模型GLM
   线性指数族（LED）的分布形式为$$f(y;\theta,\phi)=exp[\frac{y\theta-b(\theta)}{\phi}+S(y,\phi)]$$，其中$$\mu=E(y)=b^{'}(\theta)$$，$$\sigma^2=Var(y)={\phi}b^{''}(\theta)$$

   GLM：y的分布属于LED并且有链接函数$$g(\mu)=\vec{x}\cdot\vec{\beta}=\beta_0+\beta_1x_1+\cdots+\beta_kx_k$$（通常取$$g(\mu)=\theta$$）。容易看出线性回归也是GLM的一种，y满足正态分布属于LED，并且$$g(\mu)=\theta=\mu$$

   系数$$\vec{\beta}$$的求解使用最大似然估计（MLE），取$$g(\mu)=\theta$$，则Log Likelihood $$l(\vec{\beta})=\sum_{i=1}^n[\frac{y_i\theta_i-b(\theta_i)}{\phi_i}+S(y_i,\phi_i)]=\sum_{i=1}^n[\frac{y_i\vec{x_{(i)}}\cdot\vec{\beta}-b(\vec{x_{(i)}}\cdot\vec{\beta})}{\phi_i}+S(y_i,\phi_i)]$$，求解方程$$\frac{\partial{l}}{\partial{\beta}}=\sum_{i=1}^n[\frac{y_i-b^{'}(\vec{x_{(i)}}\cdot\vec{\beta})}{\phi_i}]\vec{x_{(i)}}^T=\sum_{i=1}^n\frac{y_i-\hat{\mu}_i}{\phi_i}\begin{bmatrix}\begin{smallmatrix}1\\x_{(i)1}\\\vdots\\x_{(i)k}\end{smallmatrix}\end{bmatrix}=0$$（注：$$\begin{bmatrix}\begin{smallmatrix}\vec{x_{(1)}}\\\vec{x_{(2)}}\\\vdots\\\vec{x_{(n)}}\end{smallmatrix}\end{bmatrix}=\begin{bmatrix}\begin{smallmatrix} 1&x_{(1)1}&x_{(1)2}&\cdots&x_{(1)k} \\ 1&x_{(2)1}&x_{(2)2}&\cdots&x_{(2)k} \\ \vdots&\vdots&\vdots&\cdots&\vdots \\ 1&x_{(n)1}&x_{(n)2}&\cdots&x_{(n)k} \end{smallmatrix}\end{bmatrix}$$）

   模型拟合度：
   + Global Measure
     + Deviance $$D=2(l_{sat}-l)$$

        Saturated Model：将$$l(\vec{\beta})$$改写成关于$$\mu_1,\mu_2,\cdots,\mu_n$$的表达形式，$$\mu_i\gets{y_i}$$
     + Pseudo-R<sup>2</sup> $$R_p^2=\frac{l-l_{iid}}{l_{sat}-l_{iid}}$$

        IID Model：将$$l(\vec{\beta})$$改写成关于$$\mu_1,\mu_2,\cdots,\mu_n$$的表达形式，$$\mu_i\gets{\bar{y}}$$
   + Local Measure
     + Pearson Residual $$r_i=\frac{y_i-\hat{\mu}_i}{\sqrt{\hat{Var}(y_i)}}$$，$$i=1,2,\cdots,n$$
     + Deviance Residual $$d_i=sign(y_i-\hat{\mu}_i)\sqrt{2[ln(f(y_i;\theta_i^{sat}))-ln(f(y_i;\hat{\theta}_i))]}$$（容易看出$$D=\sum_{i=1}^nd_i^2$$）
   + Variable Selection Measure
     + $$AIC=-2l+2p$$，$$BIC=-2l+p\cdot{ln(n)}$$，其中$$p$$为参数个数，$$l$$为Log Likelihood，$$n$$为训练样本个数

   系数的假设检验（Likelihood Ratio Test）:
   + 不失一般性，令H<sub>0</sub>：$$\beta_1=\beta_2=\cdots=\beta_r=0$$  vs  H<sub>1</sub>：No constraints on $$\vec{\beta}$$

      令$$l_0$$表示在假设H<sub>0</sub>下的Log Likelihood，$$LRT=2(l-l_0)$$在假设H<sub>0</sub>下满足自由度为$$r$$的$$\chi^2$$分布$$\chi_r^2$$（容易看出$$D\sim{\chi^2_{n-(k+1)}}$$），因此若$$LRT>\chi^2_{r,\alpha}$$，则在显著性水平为$$\alpha$$时拒绝原假设H<sub>0</sub>

      若H<sub>0</sub>：$$\beta_1=\beta_2=\cdots=\beta_k=0$$，则$$LRT=2(l-l_{iid})\sim{\chi_k^2}$$

   下面介绍两个比较常用的GLM：
   1. Logistic Regression
      + Bernoulli分布$$f(y;\pi)=\pi^y(1-\pi)^{1-y}$$（其中$$y\in{\{0,1\}}$$），容易得出$$\mu=E(y)=\pi$$，$$\phi=1$$，$$\theta=ln\frac{\pi}{1-\pi}$$，$$b(\theta)=ln(1+e^{\theta})$$
      + 令$$\theta=g(\mu)=\vec{x}\cdot\vec{\beta}$$，则$$\mu=\pi=\frac{1}{1+e^{-\theta}}=\frac{1}{1+e^{-\vec{x}\cdot\vec{\beta}}}$$，$$\vec{\beta}$$的最大似然估计值满足$$\sum_{i=1}^n(y_i-\hat{\pi}_i)\vec{x_{(i)}}^T=\sum_{i=1}^n(y_i-\frac{1}{1+e^{-\vec{x_{(i)}}\cdot\hat{\vec{\beta}}}})\begin{bmatrix}\begin{smallmatrix}1\\x_{(i)1}\\\vdots\\x_{(i)k}\end{smallmatrix}\end{bmatrix}=0$$
      + Deviance $$D=2\sum_{i=1}^n[y_iln\frac{y_i}{\hat{\pi}_i}+(1-y_i)ln\frac{1-y_i}{1-\hat{\pi}_i}]$$
      + 因为$$\hat{\vec{\beta}}$$是最大似然估计值，所以$$\hat{Cov}(\hat{\vec{\beta}})=I(\hat{\vec{\beta}})^{-1}$$，其中$$I(\hat{\vec{\beta}})=-\frac{\partial^2}{\partial\vec{\beta}(\partial\vec{\beta})^T}l(\vec{\beta})\mid_{\vec{\beta}=\hat{\vec{\beta}}}=\sum_{i=1}^n\frac{e^{\vec{x_{(i)}}\cdot\hat{\vec{\beta}}}}{(1+e^{\vec{x_{(i)}}\cdot\hat{\vec{\beta}}})^2}\vec{x_{(i)}}^T\vec{x_{(i)}}=\sum_{i=1}^n\frac{e^{\vec{x_{(i)}}\cdot\hat{\vec{\beta}}}}{(1+e^{\vec{x_{(i)}}\cdot\hat{\vec{\beta}}})^2}\begin{bmatrix}\begin{smallmatrix}1\\x_{(i)1}\\\vdots\\x_{(i)k}\end{smallmatrix}\end{bmatrix}\begin{bmatrix}\begin{smallmatrix}1&x_{(i)1}&\cdots&x_{(i)k}\end{smallmatrix}\end{bmatrix}$$为Fisher信息矩阵
      + 若$$y\in{\{1,2,\cdots,c\}}$$且为nominal类型，则可令$$ln\frac{\pi_j}{\pi_c}=\vec{x}\cdot\vec{\beta}_j$$，$$j=1,2,\cdots,c$$并且$$\vec{\beta}_c=\vec{0}$$，容易看出$$\pi_j=\frac{e^{\vec{x}\cdot\vec{\beta}_j}}{\sum_{m=1}^c{e^{\vec{x}\cdot\vec{\beta}_m}}}$$，$$j=1,2,\cdots,c$$
      + 若$$y\in{\{1,2,\cdots,c\}}$$且为ordinal类型，则可令$$ln\frac{\tau_j}{1-\tau_j}=ln\frac{\pi_1+\pi_2+\cdots+\pi_j}{\pi_{j+1}+\pi_{j+2}+\cdots+\pi_c}=\vec{x}\cdot\vec{\beta}_j$$，$$j=1,2,\cdots,c-1$$，容易看出
      $$\tau_j=\frac{1}{1+e^{-\vec{x}\cdot\vec{\beta}_j}}$$，$$j=1,2,\cdots,c-1$$。还可以使用一个简化的模型求解，记$$\vec{\beta}_j=\begin{bmatrix}\begin{smallmatrix}\beta_{j0} \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_k\end{smallmatrix}\end{bmatrix}$$，其中$$\beta_1,\beta_2,\cdots,\beta_k$$与$$j$$无关，即$$ln\frac{\tau_j}{1-\tau_j}=\beta_{j0}+\beta_1x_1+\beta_2x_2+\cdots+\beta_kx_k$$
   2. Poisson Regression
      + Poisson分布$$f(y;\lambda)=\frac{\lambda^y}{y!}e^{-\lambda}$$，容易得出$$\mu=E(y)=\lambda$$，$$\phi=1$$，$$\theta=ln\lambda$$，$$b(\theta)=e^\theta$$
      + 令$$\theta=g(\mu)=\vec{x}\cdot\vec{\beta}$$，则$$\mu=\lambda=e^{\vec{x}\cdot\vec{\beta}}$$，$$\vec{\beta}$$的最大似然估计值满足$$\sum_{i=1}^n(y_i-\hat{\lambda}_i)\vec{x_{(i)}}^T=\sum_{i=1}^n(y_i-e^{\vec{x_{(i)}}\cdot\hat{\vec{\beta}}})\begin{bmatrix}\begin{smallmatrix}1\\x_{(i)1}\\\vdots\\x_{(i)k}\end{smallmatrix}\end{bmatrix}=0$$，此外Fisher信息矩阵$$I(\hat{\vec{\beta}})=\sum_{i=1}^ne^{\vec{x_{(i)}}\cdot\hat{\vec{\beta}}}\vec{x_{(i)}}^T\vec{x_{(i)}}=e^{\vec{x_{(i)}}\cdot\hat{\vec{\beta}}}\begin{bmatrix}\begin{smallmatrix}1\\x_{(i)1}\\\vdots\\x_{(i)k}\end{smallmatrix}\end{bmatrix}\begin{bmatrix}\begin{smallmatrix}1&x_{(i)1}&\cdots&x_{(i)k}\end{smallmatrix}\end{bmatrix}$$
      + Goodness Of Fit Statistics
         + Likelihood Ratio Test：Deviance $$D=2\sum_{i=1}^ny_iln\frac{y_i}{\hat{\lambda}_i}-2\sum_{i=1}^ny_i+2\sum_{i=1}^n\hat{\lambda}_i=2\sum_{i=1}^ny_iln\frac{y_i}{\hat{\lambda}_i}\sim{\chi^2_{n-(k+1)}}$$
         + Pearson $$\chi^2$$ Test：$$X^2=\sum_{i=1}^n(\frac{y_i-\hat{\lambda}_i}{\sqrt{\hat{\lambda}_i}})^2\sim{\chi^2_{n-(k+1)}}$$
         + 上述两个检验在样本数趋于无穷大时是等价的
