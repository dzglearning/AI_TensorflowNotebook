#coding:utf-8
#带利润条件、自定义损失函数预测

#生成数据集
import tensorflow as tf
import numpy as np
import time
batchsize = 8
SEED = 12306
lr_rate = 0.001
COST = 9
PROFIT = 1
#成本9元，利润1元

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

#输入、参数和输出，前向传播
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

#自定义loss
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ - y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.0008).minimize(loss)
#train_step = tf.train.MomentumOptimizer(lr_rate,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss)

#生成会话，训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 3000
    timeS = time.clock()
    for i in range(STEPS):
        start = (i*batchsize) % 32
        end = (i*batchsize) % 32 + batchsize
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 400 == 0:
            print "训练 %d 轮后, w1 是: " % (i)
            print sess.run(w1), "\n"
    print "最终 w1 是: \n", sess.run(w1)
    timeE = time.clock()
    print "时间为：\n", timeE-timeS,"s\n"
