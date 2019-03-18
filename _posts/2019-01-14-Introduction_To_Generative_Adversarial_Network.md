---
layout: post
title: "生成对抗网络GAN介绍"
tags: [深度学习]
date: 2019-01-14
---
### 生成对抗网络GAN由生成器和判别器两部分组成

判别器是常规的神经网络分类器，一半时间判别器接收来自训练数据中的真实图像，另一半时间收到来自生成器中的虚假图像。训练判别器使得对于真实图像，它输出的概率值接近1，而对于虚假图像则接近0

生成器与判别器正好相反，通过训练，它输出判别器赋值概率接近1的图像。生成器需要产生更加真实的输出，从而欺骗判别器

在GAN中要同时使用两个优化器，分别用来最小化判别器和生成器的损失

### Batch Normalization

Batch Normalization是DCGAN(Deep Covolutional GAN)中常用的技术，它可以使网络训练得更快，允许更大的学习率，使更多的激活函数变得有效，并且使得参数更易初始化

BN一般用于激活函数使用之前，对每个输出节点，记第$$i$$个训练样本在该节点的输出为$$x_i$$，批次均值$${\mu}_B=\frac{1}{m}\sum_{i=1}^{m}x_i$$，批次方差$${\sigma}_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-{\mu}_B)^2$$

则$$\hat{x}_i=\frac{x_i-{\mu}_B}{\sqrt{\sigma_B^2+\epsilon}}$$，$$\epsilon$$是一个很小的正值（例如0.001），BN的输出为$$y_i= \gamma\hat{x}_i+\beta$$，$$\gamma$$和$$\beta$$均为可训练参数

同时用$${\mu}_B$$和$${\sigma}_B^2$$更新总体的均值和方差，总体均值和方差在检验网络和进行预测时使用，$${\mu}_P=\tau{\mu}_P+(1-\tau){\mu}_B$$和$${\sigma}_P^2=\tau{\sigma}_P^2+(1-\tau){\sigma}_B^2$$，$${\mu}_P$$和$${\sigma}_P^2$$的初始值为0和1，$$\tau$$可取为0.99

### DCGAN应用示例

使用的数据集为[the Street View House Numbers(SVHN) dataset](http://ufldl.stanford.edu/housenumbers/)

1. 数据处理
```python
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
### 读取数据
data_dir = 'data/'
trainset = loadmat(data_dir + 'svhntrain_32x32.mat')
testset = loadmat(data_dir + 'svhntest_32x32.mat')
#the same scale as tanh activation function
def scale(x, feature_range=(-1, 1)):
       # scale to (0, 1)
       x = ((x - x.min())/(255 - x.min()))    
       # scale to feature_range
       min, max = feature_range
       x = x * (max - min) + min
       return x
### 数据准备
class Dataset:
       def __init__(self, train, test, val_frac=0.5, shuffle=False, scale_func=None):
           split_idx = int(len(test['y'])*(1 - val_frac))
           self.test_x, self.valid_x = test['X'][:,:,:,:split_idx], test['X'][:,:,:,split_idx:]
           self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
           self.train_x, self.train_y = train['X'], train['y']
           ###(32,32,3,:) to (:,32,32,3)    
           self.train_x = np.rollaxis(self.train_x, 3)
           self.valid_x = np.rollaxis(self.valid_x, 3)
           self.test_x = np.rollaxis(self.test_x, 3)        
           if scale_func is None:
               self.scaler = scale
           else:
               self.scaler = scale_func
           self.shuffle = shuffle        
       def batches(self, batch_size):
           if self.shuffle:
               idx = np.arange(len(self.train_x))
               np.random.shuffle(idx)
               self.train_x = self.train_x[idx]
               self.train_y = self.train_y[idx]        
           n_batches = len(self.train_y)//batch_size
           for ii in range(0, len(self.train_y), batch_size):
               x = self.train_x[ii:ii+batch_size]
               y = self.train_y[ii:ii+batch_size]            
               yield self.scaler(x), y
### 创建数据集
dataset = Dataset(trainset, testset)
```

2. 搭建网络
```python
### Input
def model_inputs(real_dim, z_dim):
       inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
       inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')    
       return inputs_real, inputs_z
### Generator
def generator(z, output_dim, reuse=False, alpha=0.2, training=True):
       with tf.variable_scope('generator', reuse=reuse):
           x1 = tf.layers.dense(z, 4*4*512) #First fully connected layer  
           x1 = tf.reshape(x1, (-1, 4, 4, 512)) #Reshape it to start the convolutional stack
           x1 = tf.layers.batch_normalization(x1, training=training)
           x1 = tf.maximum(alpha * x1, x1) #leaky relu, 4x4x512 now
           # transpose convolution > batch norm > leaky ReLU     
           x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding='same') #with zero padding
           x2 = tf.layers.batch_normalization(x2, training=training)
           x2 = tf.maximum(alpha * x2, x2) #8x8x256 now
           # transpose convolution > batch norm > leaky ReLU
           x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding='same')
           x3 = tf.layers.batch_normalization(x3, training=training)
           x3 = tf.maximum(alpha * x3, x3) #16x16x128 now    
           # output layer
           logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same') #32x32x3 now          
           out = tf.tanh(logits)        
           return out
### Discriminator
def discriminator(x, reuse=False, training=True, alpha=0.2):
       with tf.variable_scope('discriminator', reuse=reuse):  
           x1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same') #Input layer is 32x32x3
           relu1 = tf.maximum(alpha * x1, x1) #16x16x64
           # convolution > batch norm > leaky ReLU
           x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same')
           bn2 = tf.layers.batch_normalization(x2, training=training)
           relu2 = tf.maximum(alpha * bn2, bn2) #8x8x128
           # convolution > batch norm > leaky ReLU
           x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
           bn3 = tf.layers.batch_normalization(x3, training=training)
           relu3 = tf.maximum(alpha * bn3, bn3) #4x4x256
           # Flatten it
           flat = tf.reshape(relu3, (-1, 4*4*256))
           logits = tf.layers.dense(flat, 1)
           out = tf.sigmoid(logits)      
           return out, logits
### Create GAN and Compute Model Loss
### input_real: Images from the real dataset
### input_z: Z input(noise)
### out_channel_dim: The number of channels in the output image
def model_loss(input_real, input_z, output_dim, training=True, alpha=0.2, smooth=0.1):
       g_model = generator(input_z, output_dim, alpha=alpha, training=training)
       d_model_real, d_logits_real = discriminator(input_real, training=training, alpha=alpha)
       # reuse=True: reuse the variables instead of creating new ones if we build the graph again
       d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, training=training, alpha=alpha)
       # real and fake loss
       d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)*(1-smooth)) #label smoothing
       d_loss_real = tf.reduce_mean(d_loss_real)
       d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake))
       d_loss_fake = tf.reduce_mean(d_loss_fake)
       ### discriminator and generator loss
       d_loss = d_loss_real + d_loss_fake
       g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake))
       g_loss = tf.reduce_mean(g_loss)
       return d_loss, g_loss, g_model
### Optimizer
### beta1: The exponential decay rate for the 1st moment in the optimizer
def model_opt(d_loss, g_loss, learning_rate, beta1):
       # Get weights and bias to update
       t_vars = tf.trainable_variables()
       d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
       g_vars = [var for var in t_vars if var.name.startswith('generator')]
       # Optimize
       with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): #update population mean and variance
           d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
           g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
       return d_train_opt, g_train_opt
### Final GAN
class GAN:
       def __init__(self, real_size, z_size, learning_rate, alpha=0.2, smooth=0.1, beta1=0.5):
           tf.reset_default_graph()      
           self.input_real, self.input_z = model_inputs(real_size, z_size)
           self.training = tf.placeholder_with_default(True, (), "train_status")        
           self.d_loss, self.g_loss, self.samples = model_loss(self.input_real, self.input_z, real_size[2], \
                                                               training=self.training, alpha=alpha, smooth=smooth)      
           self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)
```

3. 训练网络
```python
def train(net, dataset, epochs, batch_size, print_every=10, show_every=100):
       saver = tf.train.Saver()
       sample_z = np.random.uniform(-1, 1, size=(72, z_size)) #samples for generator to generate(for plotting)
       samples, losses = [], []
       steps = 0
       with tf.Session() as sess:
           sess.run(tf.global_variables_initializer())
           for e in range(epochs):
               for x, y in dataset.batches(batch_size):
                   steps += 1
                   ### sample random noise for Generator
                   batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                   ### run optimizers
                   _, _ = sess.run([net.d_opt, net.g_opt], feed_dict={net.input_real:x, net.input_z:batch_z})
                   ### get the losses and print them out
                   if steps % print_every == 0:  
                       train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: x})
                       train_loss_g = net.g_loss.eval({net.input_z: batch_z})
                       print("Epoch {}/{}...".format(e+1, epochs), \
                             "Discriminator Loss: {:.4f}...".format(train_loss_d), \
                             "Generator Loss: {:.4f}".format(train_loss_g))                     
                       losses.append((train_loss_d, train_loss_g)) #save losses to view after training
                   ### save generated samples
                   if steps % show_every == 0:
                       # training=False: the batch normalization layers will use the population statistics rather than the batch statistics
                       gen_samples = sess.run(net.samples, feed_dict={net.input_z: sample_z, net.training: False})
                       samples.append(gen_samples)                       
           saver.save(sess, './checkpoints/generator.ckpt')
       with open('samples.pkl', 'wb') as f:
           pkl.dump(samples, f)
       return losses, samples
```

4. 最终结果并可视化
```python
### Hyperparameters
real_size = (32,32,3)
z_size = 100
learning_rate = 0.0002
batch_size = 128
epochs = 25
alpha = 0.2
smooth = 0.1
beta1 = 0.5
### Create and Train the network
net = GAN(real_size, z_size, learning_rate, alpha=alpha, smooth=smooth, beta1=beta1)
losses, samples = train(net, dataset, epochs, batch_size)
### Visualize
def view_samples(sample, nrows, ncols, figsize=(5,5)): #the number of the sample=nrows*ncols
       fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=True, sharex=True)
       for ax, img in zip(axes.flatten(), sample):
           ax.axis('off')
           img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
           ax.set_adjustable('box-forced')
           im = ax.imshow(img, aspect='equal')   
       plt.subplots_adjust(wspace=0, hspace=0)
       return fig, axes
view_samples(samples[-1], 6, 12, figsize=(10,5))
```

![img](/img/svhn.png)
