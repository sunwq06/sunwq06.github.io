---
layout: post
title: "PCA与LDA降维介绍"
tags: [机器学习]
date: 2019-02-01
---

### PCA(主成分分析)

PCA是一种无监督降维方式，它将数据投影到一组互相正交的loading vectors(principal axes)之上，并保证投影后的点在新的坐标轴上的方差最大

1. 记数据集$$X=\begin{bmatrix}\begin{smallmatrix}\vec{x_1}\\\vec{x_2}\\\vdots\\\vec{x_n}\end{smallmatrix}\end{bmatrix}$$为n行p列的矩阵（n个数据，每个数据p维），特征均值为$$\vec{\mu}=(\mu_1, \mu_2, .., \mu_p)$$，数据与均值的差异可表示为$$\tilde{X}=\begin{bmatrix}\begin{smallmatrix}\vec{x_1}-\vec{\mu}\\\vec{x_2}-\vec{\mu}\\\vdots\\\vec{x_n}-\vec{\mu}\end{smallmatrix}\end{bmatrix}$$

2. 假设需求解m个loading vector $$\vec{\phi}_1,\vec{\phi}_2,...,\vec{\phi}_m$$，$${m}\leq{min(n-1,p)}$$，且需满足$$\vec{\phi}_i^T\vec{\phi}_i=1$$以及$$\vec{\phi}_i^T\vec{\phi}_j=0, i\neq{j}$$

3. $$X$$在$$\vec{\phi}_1$$上的投影为$$X\vec{\phi}_1$$，特征均值的投影为$$\vec{\mu}\cdot\vec{\phi}_1$$，则投影后数据与均值的差异可表示为$$\tilde{X}\vec{\phi}_1$$，投影后的方差为$$\vec{\phi}_1^T\tilde{X}^T\tilde{X}\vec{\phi}_1$$（省略了系数$$\frac{1}{n}$$）

4. 记$$Q=\tilde{X}^T\tilde{X}$$，$$Q$$即为数据集X的协方差矩阵。将$$Q$$进行特征值分解$$Q=V\Lambda{V^T}$$，其中$$\Lambda$$为对角矩阵，对角线上的元素为特征值（不失一般性，这里令其按从大到小的顺序排列）；$$V=\begin{bmatrix}\begin{smallmatrix}\vec{v_1}&\vec{v_2}&\cdots&\vec{v_p}\end{smallmatrix}\end{bmatrix}$$为正交矩阵，它的列为对应的特征向量

5. 投影后的方差可以写成$$\vec{\phi}_1^TV\Lambda{V^T}\vec{\phi}_1=\vec{a}_1^T\Lambda\vec{a}_1=\sum_{i=1}^p\lambda_ia_{1i}^2$$，因为$$\sum_{i=1}^pa_{1i}^2=\vec{a}_1^T\vec{a}_1=\vec{\phi}_1^TVV^T\vec{\phi}_1=\vec{\phi}_1^T\vec{\phi}_1=1$$，所以方差的最大值为$$\lambda_1$$，并且仅当$$\vec{\phi}_1=\vec{v}_1$$时取到

6. $$X$$在$$\vec{\phi}_2$$上投影后的方差可以表示为$$\sum_{i=1}^p\lambda_ia_{2i}^2$$（同上步类似，$$\vec{a}_2=V^T\vec{\phi}_2$$ ，$$\sum_{i=1}^pa_{2i}^2=1$$），又因为$$a_{21}=\vec{v}_1^T\vec{\phi}_2=\vec{\phi}_1^T\vec{\phi}_2=0$$，所以方差的最大值为$$\lambda_2$$，并且仅当$$\vec{\phi}_2=\vec{v}_2$$时取到

7. 对于$$\vec{\phi}_i, i=3,...,m$$可以按上述步骤依次求得，方差的最大值为$$\lambda_i$$，并且仅当$$\vec{\phi}_i=\vec{v}_i$$时取到

8. 实际应用中首先将数据集$$X$$进行标准化（减去特征均值并除以特征标准差），此时协方差矩阵$$Q=X^TX$$，对$$X$$进行SVD分解，$$X=USV$$，其中$$U$$为n行n列的正交矩阵，列向量为$$XX^T$$的特征向量；$$V$$为p行p列的正交矩阵，列向量为$$X^TX$$的特征向量（即同将$$Q$$进行特征值分解得到的$$V$$）；$$S$$为n行p列的矩阵且非对角线上的元素为0，对角线上的元素$$s_{ii}=\sqrt{\lambda_i}$$

### LDA(线性判别分析)

LDA是一种有监督降维方式，假设数据集$$X$$共分为$$K$$个类，需保证投影后的点在新的坐标轴上类内离散度尽可能小，同时类间离散度尽可能大

1. 记$$\vec{\mu}_k$$为第k个类的特征均值，$$\vec{\mu}$$为总体的特征均值，则特征均值的估计值$$\hat{\vec{\mu}}_k=\frac{\sum_{i\in{class}\ {k}}\vec{x}_i}{n_k}$$，$$\hat{\vec{\mu}}=\frac{\sum_{i=1}^n\vec{x}_i}{n}$$
2. 记$$C_k$$为第k个类的协方差矩阵，$$C$$为总体的协方差矩阵，LDA假设$$C_1=C_2=\cdots=C_K=C$$，则协方差矩阵的估计值$$\hat{C}=\sum_{k=1}^K\sum_{i\in{class}\ {k}}(\vec{x}_i-\hat{\vec{\mu}}_k)^T(\vec{x}_i-\hat{\vec{\mu}}_k)$$（省略了系数$$\frac{1}{n-K}$$）
3. 假设投影坐标轴为$$\vec{\phi}$$，第k类中数据与均值的差异可表示为$$\tilde{X}_k=\begin{bmatrix}\begin{smallmatrix}\vec{x}_{k_1}-\hat{\vec{\mu}}_k\\\vec{x}_{k_2}-\hat{\vec{\mu}}_k\\\vdots\\\vec{x}_{k_{n_k}}-\hat{\vec{\mu}}_k\end{smallmatrix}\end{bmatrix}$$，第k类的数据投影后的离散度可表示为$$\vec{\phi}^T\tilde{X}_k^T\tilde{X}_k\vec{\phi}$$，$$K$$个类的类内离散度之和为$$\vec{\phi}^T\sum_{k=1}^K\tilde{X}_k^T\tilde{X}_k\vec{\phi}=\vec{\phi}^T\hat{C}\vec{\phi}$$
4. 由PCA的第三步可以看出投影后数据的总体离散度为$$\vec{\phi}^T\tilde{X}^T\tilde{X}\vec{\phi}$$，其中$$\tilde{X}=\begin{bmatrix}\begin{smallmatrix}\vec{x_1}-\hat{\vec{\mu}}\\\vec{x_2}-\hat{\vec{\mu}}\\\vdots\\\vec{x_n}-\hat{\vec{\mu}}\end{smallmatrix}\end{bmatrix}$$，则类间离散度可以表示为总体与类内离散度之差，即$$\vec{\phi}^T[\tilde{X}^T\tilde{X}-\hat{C}]\vec{\phi}=\vec{\phi}^T[\sum_{k=1}^Kn_k(\hat{\vec{\mu}}-\hat{\vec{\mu}}_k)^T(\hat{\vec{\mu}}-\hat{\vec{\mu}}_k)]\vec{\phi}=\vec{\phi}^TB\vec{\phi}$$
5. 为了使类内离散度尽可能小，同时类间离散度尽可能大，先将类内离散度转化为常数，然后只考虑类间离散度。因此首先进行一个空间变换，使得新空间上的协方差矩阵变为单位矩阵，对$$\hat{C}$$进行特征值分解$$\hat{C}=UDU^T$$，记$$W=UD^{-1/2}$$为空间变换矩阵，新空间上的数据集变为$$X^*=XW$$。假设在新空间上的投影坐标轴为$$\vec{\phi}^*$$，容易看出在新空间上的类内离散度为$$\vec{\phi}^{*T}W^T\hat{C}W\vec{\phi}^*=\vec{\phi}^{*T}W^T\hat{C}W\vec{\phi}^*=\vec{\phi}^{*T}I\vec{\phi}^*=1$$
6. 新空间上的类间离散度变为$$\vec{\phi}^{*T}W^TBW\vec{\phi}^*$$，此时可以参照PCA的做法，在新空间上依次寻找互相正交的坐标轴，使得新空间上的类间离散度最大。对$$W^TBW$$进行特征值分解$$W^TBW=V\Lambda{V^T}$$，容易看出$$\vec{\phi}^{*}_i=\vec{v}_i$$，$$i=1,2,\cdots,m$$，$$m\leq{K-1}$$（证明过程见PCA的5-7步）
7. 综上所述，最终求得的坐标轴$$\vec{\phi}_i=W\vec{\phi}^{*}_i$$，$$i=1,2,\cdots,m$$
8. 对于$$\vec{\phi}^{*}_i$$，有$$W^TBW\vec{\phi}^{*}_i=\lambda_i\vec{\phi}^{*}_i$$，等式两边同时左乘W，有$$WW^TBW\vec{\phi}^{*}_i=\lambda_iW\vec{\phi}^{*}_i$$，即$$UD^{-1}U^TB\vec{\phi}_i=\hat{C}^{-1}B\vec{\phi}_i=\lambda_i\vec{\phi}_i$$。因此上述步骤等价于直接求解$$\hat{C}^{-1}B$$的特征值和特征向量（注意此时的特征向量 $$\vec{\phi}$$不是单位向量$$\vec{\phi}^T\vec{\phi}=1$$，而是需满足$$[W^{-1}\vec{\phi}]^TW^{-1}\vec{\phi}=\vec{\phi}^T\hat{C}\vec{\phi}=1$$），将此时对应的特征值按从大到小排列取前m个特征值和特征向量
9. 参考文献: [The Elements of Statistical Learning(2nd Edition)](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf) Section 4.3.3  
