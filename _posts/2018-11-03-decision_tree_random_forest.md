---
layout: post
title: "CART决策树和随机森林"
tags: [机器学习]
date: 2018-11-03
---

CART

1. 分裂规则
  + 将现有节点的数据分裂成两个子集，计算每个子集的gini index
  + 子集的Gini index: $$gini_{child}=\sum_{i=1}^K p_{ti} \sum_{i' \neq i} p_{ti'}=1-\sum_{i=1}^K p_{ti}^2$$ ， 其中K表示类别个数，$$p_{ti}$$表示分类为i的样本在子集中的比例，gini index可以理解为该子集中的数据被错分成其它类别的期望损失
  + 分裂后的Gini index: $$gini_s= \frac{N_1}{N}gini_{child_1}+\frac{N_2}{N}gini_{child_2}$$ ，其中N为分裂之前的样本数，$$N_1$$和$$N_2$$为分裂之后两个子集的样本数
  + 选取使得$$gini_s$$最小的特征和分裂点进行分裂


2. 减少过拟合
  + 设置树的最大深度(max_depth in sklearn.tree.DecisionTreeClassifier)
  + 设置每个叶子节点的最少样本个数(min_samples_leaf in sklearn.tree.DecisionTreeClassifier)
  + 剪枝


3. 样本均衡问题
  + 若样本的类别分布极不均衡，可对每个类i赋予一个权重$$w_i$$, 样本较少的类赋予较大的权重(class_weight in sklearn.tree.DecisionTreeClassifier)，此时算法中所有用到样本类别个数的地方均转换成类别的权重和。例如$$p_{ti}=\frac{w_{i}N_i}{\sum_{i=1}^K w_{i}N_i}$$ ，其中$$N_i$$为在子集中类别为i的样本数; $$gini_s=\frac{weightsum(N_1)}{weightsum(N)}gini_{child_1}+\frac{weightsum(N_2)}{weightsum(N)}gini_{child_2}$$


4. 回归问题
  + 和分类问题相似，只是分裂规则中的$$gini_{child}$$变为了mean squared error，即$$MSE_{child}=\frac{1}{N_{child}}\sum_{i \in child}(y_i-\bar{y}_{child})^2$$


Random Forest

1. 随机性
  + 在每次建立新树的时候通过bootstrap方法从N个训练样本中有放回地随机选出N个新的样本(bootstrap in sklearn.ensemble.RandomForestClassifier)
  + 在每次分裂的时候从所有特征中随机选取部分特征进行查找(max_features in sklearn.ensemble.RandomForestClassifier)


2. 样本均衡问题
  + 同CART一样，样本较少的类赋予较大的权重(class_weight in sklearn.ensemble.RandomForestClassifier)
  + 需要注意的是权重对于bootstrap的使用并没有影响，即bootstrap方法始终是等概率地从N个样本中选择，sklearn中的源码如下
  ```python
  if forest.bootstrap:
             n_samples = X.shape[0]
             if sample_weight is None:
                 curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
             else:
                 curr_sample_weight = sample_weight.copy() #已经包含了class_weight设为'balanced'或dict类型时的类别权重

             indices = _generate_sample_indices(tree.random_state, n_samples) #bootstrap
             sample_counts = np.bincount(indices, minlength=n_samples)
             curr_sample_weight *= sample_counts #根据新的样本集合中每个原始样本的个数来调整样本权重
             ### 根据类别权重调整样本权重
             if class_weight == 'subsample':
                 with catch_warnings():
                     simplefilter('ignore', DeprecationWarning)
                     curr_sample_weight *= compute_sample_weight('auto', y, indices)
             elif class_weight == 'balanced_subsample':
                 curr_sample_weight *= compute_sample_weight('balanced', y, indices)

             tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
  else:
             tree.fit(X, y, sample_weight=sample_weight, check_input=False)
  ```


3. OOB(out-of-bag estimate)
  + 对每一个训练样本zi=(xi, yi)，使用没有选中该样本的那些树构建该样本的随机森林预测
  + 计算所有训练样本的预测准确率(oob_score_ in sklearn.ensemble.RandomForestClassifier)
  + 很明显，只有bootstrap设为True时OOB才是有效的


4. 特征重要性
  + 在CART构建过程中使用某特征进行分裂导致的gini系数的总的减少越多，那么认为该特征的重要性就越大
  + 随机森林中的特征重要性是各个决策树中的重要性总和或平均(feature_importances_ in sklearn.ensemble.RandomForestClassifier)
