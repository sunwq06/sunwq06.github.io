---
layout: post
title: "机器学习应用示例：多类别多标签问题"
tags: [机器学习]
date: 2018-12-14
---

本文使用的数据（[下载地址](https://pan.baidu.com/s/1w8MI70oAwK_3knEjdzW1aQ)）是一个多类别多标签分类问题，具体介绍和问题描述参考[此链接](https://www.drivendata.org/competitions/4/box-plots-for-education/page/15)

1. 建立拆分训练集和测试集的函数
```python
import numpy as np
import pandas as pd
from warnings import warn
### Takes a label matrix 'y' and returns the indices for a sample with size
### 'size' if 'size' > 1 or 'size' * len(y) if 'size' <= 1.
### The sample is guaranteed to have > 'min_count' of each label.
def multilabel_sample(y, size=1000, min_count=5, seed=None):
       try:
           if (np.unique(y).astype(int) != np.array([0, 1])).any():
               raise ValueError()
       except (TypeError, ValueError):
           raise ValueError('multilabel_sample only works with binary indicator matrices')
       if (y.sum(axis=0) < min_count).any():
           raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')
       if size <= 1:
           size = np.floor(y.shape[0] * size)
       if y.shape[1] * min_count > size:
           msg = "Size less than number of columns * min_count, returning {} items instead of {}."
           warn(msg.format(y.shape[1] * min_count, size))
           size = y.shape[1] * min_count #size should be at least this value for having >min_count of each label
       rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))
       if isinstance(y, pd.DataFrame):
           choices = y.index
           y = y.values
       else:
           choices = np.arange(y.shape[0])
       sample_idxs = np.array([], dtype=choices.dtype)
       # first, guarantee > min_count of each label
       for j in range(y.shape[1]):
           label_choices = choices[y[:, j] == 1]
           label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
           sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])
       sample_idxs = np.unique(sample_idxs)
       # now that we have at least min_count of each, we can just random sample
       sample_count = int(size - sample_idxs.shape[0])
       # get sample_count indices from remaining choices
       remaining_choices = np.setdiff1d(choices, sample_idxs)
       remaining_sampled = rng.choice(remaining_choices, size=sample_count, replace=False)
       return np.concatenate([sample_idxs, remaining_sampled])   
### Takes a features matrix X and a label matrix Y
### Returns (X_train, X_test, Y_train, Y_test) where all classes in Y are represented at least min_count times.     
def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):       
       index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
       test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
       test_set_mask = index.isin(test_set_idxs)
       train_set_mask = ~test_set_mask
       return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])
```

2. 定义loss metric(Logarithmic Loss metric)
```python
### 数据共对应9个标签，每个标签又有不同的类别个数
LABEL_INDICES = [range(0, 37), range(37, 48), range(48, 51), range(51, 76), range(76, 79), \
                    range(79, 82), range(82, 87), range(87, 96), range(96, 104)]
### Logarithmic Loss metric
### predicted, actual: 2D numpy array
def multi_multi_log_loss(predicted, actual, label_column_indices=LABEL_INDICES, eps=1e-15):
       label_scores = np.ones(len(label_column_indices), dtype=np.float64)
       # calculate log loss for each set of columns that belong to a label
       for k, this_label_indices in enumerate(label_column_indices):
           # get just the columns for this label
           preds_k = predicted[:, this_label_indices].astype(np.float64)
           # normalize so probabilities sum to one (unless sum is zero, then we clip)
           preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)
           actual_k = actual[:, this_label_indices]
           # shrink predictions
           y_hats = np.clip(preds_k, eps, 1 - eps)
           sum_logs = np.sum(actual_k * np.log(y_hats))
           label_scores[k] = (-1.0 / actual.shape[0]) * sum_logs
       return np.average(label_scores)        
```

3. 读取并拆分数据集
```python
### 读取并拆分数据
df = pd.read_csv('TrainingData.csv', index_col=0)
LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', \
             'Object_Type', 'Pre_K', 'Operating_Status']
NON_LABELS = [c for c in df.columns if c not in LABELS]
NUMERIC_COLUMNS = ['FTE', 'Total']
label_dummies = pd.get_dummies(df[LABELS], prefix_sep='__')
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS], label_dummies, size=0.2, seed=123)
```

4. 对数据特征进行预处理
```python
###(1) 将每行所有文本整合成一个字符串
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):   
       to_drop = set(to_drop) & set(data_frame.columns.tolist()) #Drop non-text columns that are in the df
       text_data = data_frame.drop(to_drop, axis=1)
       text_data.fillna('', inplace=True)
       # Join all text items in a row that have a space in between
       return text_data.apply(lambda x: ' '.join(x), axis=1)
###(2) 读取文本特征
from sklearn.preprocessing import FunctionTransformer
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
###(3) 读取数值特征
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)
###(4) 对文本特征按照标点和空格进行分词，并使用 1-gram 和 2-gram
###    Option 1: CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1, 2))
###              记录文本数据中出现的每个一元和二元词组，计算每行文本数据中每个一元和二元词组出现的次数
###    Option 2: HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1, 2), alternate_sign=False, norm=None, binary=False, n_features=...)  
###              若CountVectorizer生成的特征太多，用HashingVectorizer替代可以控制特征数目，同时不牺牲太多精度
###              将文本数据中每个一元和二元词组映射为一个哈希值，计算每行文本数据中每个哈希值出现的次数
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=[!"#$%&\'()*+,-./:;<=>?@[\\\\\]^_`{|}~\\s]+)'  #(?=re)表示当re也匹配成功时输出'('前面的部分
text_vectorizer = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1, 2))
###(5) 对生成的文本特征进行选择(使用卡方检验)
###    原理如下所示：
###    observation O = np.dot(y.T, X), X中的元素>=0, y中的元素为0或1, 2D matrix (num_class, num_feature)
###    expectation E = np.dot(y.mean(axis=0).T, X.sum(axis=0)), 2D matrix (num_class, num_feature)
###    卡方统计量 = ((O-E)**2/E).sum(axis=0), 1D array (num_feature,)
###    每个特征对应的卡方统计量在该特征与分类结果无关的假设条件下服从自由度为num_class-1的卡方分布
###    每个特征对应的卡方统计量越大, 该特征就越重要
from sklearn.feature_selection import chi2, SelectKBest
chi_k = 300
text_feature_selector = SelectKBest(chi2, chi_k)
###(6) 合并数值和文本特征
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer         
num_text_feature = FeatureUnion([('numeric_features', Pipeline([('selector', get_numeric_data), ('imputer', Imputer())])), \
                                    ('text_features', Pipeline([('selector', get_text_data), ('vectorizer', text_vectorizer), ('dim_red', text_feature_selector)]))])
###(7) 特征Interaction
###    同sklearn中的PolynomialFeatures，但由于CountVectorizer或HashingVectorizer得到的是稀疏矩阵，
###    不能直接用PolynomialFeatures
from itertools import combinations
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
class SparseInteractions(BaseEstimator, TransformerMixin):
       def __init__(self, degree=2, feature_name_separator='&&'):
           self.degree = degree
           self.feature_name_separator = feature_name_separator
       def fit(self, X, y=None):
           return self
       def transform(self, X):
           if not sparse.isspmatrix_csc(X):
               X = sparse.csc_matrix(X)
           if hasattr(X, "columns"):
               self.orig_col_names = X.columns
           else:
               self.orig_col_names = np.array([str(i) for i in range(X.shape[1])])
           spi = self._create_sparse_interactions(X)
           return spi
       def get_feature_names(self):
           return self.feature_names
       def _create_sparse_interactions(self, X):
           out_mat = []
           self.feature_names = self.orig_col_names.tolist()
           for sub_degree in range(2, self.degree + 1):
               for col_ixs in combinations(range(X.shape[1]), sub_degree):
                   # add name for new column
                   name = self.feature_name_separator.join(self.orig_col_names[list(col_ixs)])
                   self.feature_names.append(name)
                   # get column multiplications value
                   out = X[:, col_ixs[0]]
                   for j in col_ixs[1:]:
                       out = out.multiply(X[:, j])
                   out_mat.append(out)
           return sparse.hstack([X] + out_mat)
```

5. 对特征进行尺度变换并使用Logistic分类建立模型
```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler
pl = Pipeline([('union', num_text_feature), ('inter', SparseInteractions(degree=2)), \
                  ('scale', MaxAbsScaler()), ('clf', OneVsRestClassifier(LogisticRegression()))])
pl.fit(X_train, y_train)
predictions = pl.predict_proba(X_test)
print("Test Logloss: {}".format(multi_multi_log_loss(predictions, y_test.values)))
```

6. 该问题还可从以下几个方面继续探索
  + NLP: e.g., stop-word removal
  + Model: e.g., Random Forest
  + Numeric Preprocessing: e.g., Imputation strategies
  + Optimization: e.g., Grid Search over pipeline objects
