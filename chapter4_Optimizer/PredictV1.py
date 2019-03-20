#coding:utf-8
#酸奶销量预测产量V1

import tensorflow as tf
import numpy as np
import time		#计时比较不同优化器
batch_size = 8
Seed = 12306
lr_rate = 0.001

#生成数据集
rdm = np.random.RandomState(Seed)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]
#加入随机噪声

#定义网络的输入、参数和输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))		#一层隐含层
y = tf.matmul(x, w1)
#前向传播输出计算结果

#损失函数为MSE,反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.square(y_ - y))
#train_step = tf.train.GradientDescentOptimizer(lr_rate).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(lr_rate,0.9).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_mse)


#生成计算图，训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    TimeStart = time.clock()
    for i in range(STEPS):
        start = (i*batch_size) % 32
        end = (i*batch_size) % 32 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 400 == 0:
            print "经过 %d 轮, w1 参数为: " % (i)
            print sess.run(w1), "\n"
    print "最终参数 w1 是: \n", sess.run(w1)
    TimeEnd = time.clock()
    print "共计时间为：\n", TimeEnd-TimeStart, "s\n"

