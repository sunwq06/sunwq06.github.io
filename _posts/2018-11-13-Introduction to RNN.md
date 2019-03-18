---
layout: post
title: "循环神经网络RNN介绍"
tags: [深度学习]
date: 2018-11-13
---
#### RNN（Recurrent Neural Network）是用于处理序列数据的神经网络，它的网络结构如下图所示

<img src="/img/rnn.PNG">

该网络的计算过程可表示为$$\bar{s}_t=\Phi(\bar{x}_tW_x+\bar{s}_{t-1}W_s), \bar{s}'_t=\Phi(\bar{s}_tW_y+\bar{s}'_{t-1}W_{s'}), \bar{O}_t=\bar{s}'_tW$$，其中$$W_x,W_y,W,W_s,W_{s'}$$为权重矩阵，$$\bar{O}_t,\bar{O}_{t+1},...$$为网络的输出

权重矩阵的计算使用BPTT算法，它的本质还是BP算法，只不过要加上基于时间的反向传播，以下图一个简单的网络为例，其中$$\bar{y}_3$$表示输出，$$\bar{d}_3$$表示实际值，$$E_3$$表示损失函数

<img src="/img/rnn1.png">

根据链式求导法则:

(1)  $$\frac{\partial{E_3}}{\partial{W_y}}=\frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{W_y}}$$

(2) $$\frac{\partial{E_3}}{\partial{W_s}}=\frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{\bar{s}_3}}\frac{\partial{\bar{s}_3}}{\partial{W_s}} + \frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{\bar{s}_3}}\frac{\partial{\bar{s}_3}}{\partial{\bar{s}_2}}\frac{\partial{\bar{s}_2}}{\partial{W_s}} + \frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{\bar{s}_3}}\frac{\partial{\bar{s}_3}}{\partial{\bar{s}_2}}\frac{\partial{\bar{s}_2}}{\partial{\bar{s}_1}}\frac{\partial{\bar{s}_1}}{\partial{W_s}}$$

(3) $$\frac{\partial{E_3}}{\partial{W_x}}=\frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{\bar{s}_3}}\frac{\partial{\bar{s}_3}}{\partial{W_x}} + \frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{\bar{s}_3}}\frac{\partial{\bar{s}_3}}{\partial{\bar{s}_2}}\frac{\partial{\bar{s}_2}}{\partial{W_x}} + \frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{\bar{s}_3}}\frac{\partial{\bar{s}_3}}{\partial{\bar{s}_2}}\frac{\partial{\bar{s}_2}}{\partial{\bar{s}_1}}\frac{\partial{\bar{s}_1}}{\partial{W_x}}$$

由上述公式可以很容易看出时间步长间隔越多，在梯度计算中累乘的项数就越多，激活函数的导数相乘的次数就越多，越容易出现梯度消失现象，即当前时刻与多个时间步长之前的时刻之间的依赖关系在计算过程中被丢弃了
+ 激活函数选取relu，右侧导数恒为1，可以较好地解决梯度消失问题；但是若W没有很好地初始化，容易产生梯度爆炸问题，需使用梯度裁剪（如果梯度的范数大于某个给定值，将梯度同比收缩）解决
+ RNN网络的一些变种（例如LSTM、GRU）可以较好地解决梯度消失问题

#### 长短时记忆网络LSTM(Long Short Term Memory network)是常规RNN网络的一个变体，可以学习长时间间隔的依赖关系，以下总结主要参考[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

<img src="/img/lstm.png">

1. 一个LSTM单元有相应的cell state($$C_t$$)
2. 遗忘门(forget layer)表示前一个单元的cell state有多少进入到当前单元

   <img src="/img/lstm1.PNG">

3. 输入门(input gate)表示有多少新的信息进入当前单元

   <img src="/img/lstm2.PNG">

4. 计算当前单元的cell state

   <img src="/img/lstm3.PNG">

5. 输出门(output gate)表示当前单元的输出值

   <img src="/img/lstm4.PNG">

#### GRU(Gated Recurrent Unit)是另一个比较常用的变体，它的网络结构如下图所示

<img src="/img/gru.PNG">
