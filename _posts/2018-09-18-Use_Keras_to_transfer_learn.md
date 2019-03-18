---
layout: post
title: "使用keras进行迁移学习(Transfer Learning)"
tags: [深度学习]
date: 2018-09-18
---

利用Keras中的预训练模型进行狗品种分类的迁移学习

数据来源：[Dog Breed Data](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

+ 导入相应的包
```python
import glob
import numpy as np
from PIL import ImageFile
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras import backend as K
ImageFile.LOAD_TRUNCATED_IMAGES = True
```
+ 搭建模型（使用resnet50）
```python
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
predictions = Dense(133, activation='softmax')(x)
### 建立模型
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers: layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```
+ 数据生成器函数（训练时使用data augmentation）
```python
def data_generator(path, sz, bs, mode):
      if mode=='train':
          #zoom_range对图像内容进行缩放，但不改变图像尺寸
          datagen = ImageDataGenerator(preprocessing_function=preprocess_input, shear_range=0.1, zoom_range=0.1, \
                                       rotation_range=10, horizontal_flip=True)
          generator = datagen.flow_from_directory(path, shuffle=True, target_size=(sz, sz), batch_size=bs, class_mode='categorical')
      else:
          datagen = ImageDataGenerator() if mode=='plot' else ImageDataGenerator(preprocessing_function=preprocess_input)
          generator = datagen.flow_from_directory(path, shuffle=False, target_size=(sz, sz), batch_size=bs, \
                                                  class_mode=None if mode=='prediction' or mode=='plot' else 'categorical')
      return generator
```
+ 训练模型
```python
### 数据存放路径
PATH = "../../data/dogbreed_udacity/dogImages/"
train_data_dir = f'{PATH}train'
validation_data_dir = f'{PATH}valid'
test_data_dir = f'{PATH}test'
### 参数设置
epochs = 5
sz=224
train_batch_size=40
valid_batch_size=167
### 数据生成
### 路径下面的子文件夹自动被当作分类的类名
train_generator = data_generator(train_data_dir, sz, train_batch_size, 'train')
validation_generator = data_generator(validation_data_dir, sz, valid_batch_size, 'validation')
dog_names = [key[4:] for key in sorted(train_generator.class_indices.keys())]
### 模型训练
### When the epoch ends, the validation generator will yield validation_steps batches,
### then average the evaluation results of all batches
checkpointer = ModelCheckpoint(filepath='./weights.best.resnet50.hdf5', verbose=1, save_best_only=True)
model.fit_generator(train_generator, train_generator.n//train_batch_size, epochs=epochs, \
                      callbacks=[checkpointer], verbose=1, workers=4, \
                      validation_data=validation_generator, validation_steps=validation_generator.n//valid_batch_size)
### 放大图片尺寸，继续训练
### Starting training on small images for a few epochs, then switching to bigger images,
### and continuing training is an amazingly effective way to avoid overfitting.                      
epochs = 15
sz=299
train_generator = data_generator(train_data_dir, sz, train_batch_size, 'train')
validation_generator = data_generator(validation_data_dir, sz, valid_batch_size, 'validation')
model.fit_generator(train_generator, train_generator.n//train_batch_size, epochs=epochs, \
                      callbacks=[checkpointer], verbose=1, workers=4, \
                      validation_data=validation_generator, validation_steps=validation_generator.n//valid_batch_size)                     
```
+ 检验模型（准确率为82.66%）
```python
model.load_weights('./weights.best.resnet50.hdf5') #加载训练后的最优模型
test_batch_size = 209
test_generator = data_generator(test_data_dir, sz, test_batch_size, 'test')
loss, accuracy = model.evaluate_generator(test_generator, test_generator.n//test_batch_size)
print(accuracy)
```
+ 预测自己的图片并可视化
```python
batch_size = 64
#该路径下应包含一个子文件夹存放自己的图片(例如predict_data_dir/dogs), 若直接放到该路径会报错
predict_data_dir='./images/MY_Images'
pred_generator = data_generator(predict_data_dir, sz, batch_size, 'prediction')
#使所有图片都能被预测
pred_steps =  pred_generator.n//batch_size if pred_generator.n%batch_size==0 else \
                (pred_generator.n//batch_size)+1
### 预测图片
dog_index = np.argmax(model.predict_generator(pred_generator, pred_steps),axis=1)
dog_breed = [dog_names[i] for i in dog_index]
### 可视化
plot_batch_size = 5
plot_generator = data_generator(predict_data_dir, sz, plot_batch_size, 'plot')
numrow = plot_generator.n//plot_batch_size if plot_generator.n%plot_batch_size==0 else \
           (plot_generator.n//plot_batch_size)+1
numcol = plot_batch_size
#每行有plot_batch_size个图片
fig, axes = plt.subplots(nrows=numrow, ncols=numcol, sharex=True, sharey=True, figsize=(2*numcol, 2*numrow))
for i,batch in enumerate(plot_generator):
      if i==numrow: break
      for (j,img),ax in zip(list(enumerate(batch)),axes[i]):
          ax.set_title(dog_breed[numcol*i+j])
          ax.imshow(image.array_to_img(img))
          ax.get_xaxis().set_visible(False)
          ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
```
<img src="/img/dog_breed.png">
