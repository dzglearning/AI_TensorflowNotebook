#coding:utf-8

import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 12306
lr_rate = 0.001		#学习率

#自制数据集
rdm = np.random.RandomState(SEED)		
X = rdm.rand(32,2)
#构造数据集的标签（正确答案） 
Ycorrect = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print "X is :\n",X
print "Ycorrect is :\n",Ycorrect

#定义神经网络的输入、参数和输出,声明前向传播
x = tf.placeholder(tf.float32, shape=(None, 2))
y_c= tf.placeholder(tf.float32, shape=(None, 1))

w1= tf.Variable(tf.random_normal([2, 5], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([5, 1], stddev=1, seed=1))	 #两层，五个节点

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)		#预测输出

#损失函数和反向传播
loss_mse = tf.reduce_mean(tf.square(y-y_c)) 
train_step = tf.train.GradientDescentOptimizer(lr_rate).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(lr_rate,0.9).minimize(loss_mse)
#train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_mse)

#创建会话，训练指定轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出初始参数值
    print "w1 init is :\n", sess.run(w1)
    print "w2 init is :\n", sess.run(w2)
    print "\n"
    
    # 训练开始
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32		#确保每组数的起始点落在数据集内
        end = start + BATCH_SIZE		#截取出一个batchsize大小的数据
        sess.run(train_step, feed_dict={x: X[start:end], y_c: Ycorrect[start:end]})
        if i % 300 == 0:		#每段时间输出一下当前损失函数值
            loss_total = sess.run(loss_mse, feed_dict={x: X, y_c: Ycorrect})	#带入全部而非单组batchsize
            print("After %d training step(s), loss_mse on all data is %g" % (i, loss_total))
    
    # 训练后的参数值
    print "\n"
    print "w1 trained is :\n", sess.run(w1)
    print "w2 trained is:\n", sess.run(w2)

'''
X is :
[[0.5848066  0.58152282]
 [0.75007602 0.83827585]
 [0.26899092 0.28122624]
 [0.51703706 0.37998161]
 [0.76098976 0.00455019]
 [0.31860551 0.08109285]
 [0.0771164  0.70838084]
 [0.45619899 0.75592839]
 [0.05252395 0.02114353]
 [0.27305989 0.4266116 ]
 [0.88868806 0.61858827]
 [0.48410613 0.11710859]
 [0.53121964 0.63218361]
 [0.04409865 0.90086321]
 [0.6650902  0.58366803]
 [0.13979254 0.09371321]
 [0.47835964 0.80846277]
 [0.06778449 0.24717991]
 [0.23880038 0.13528055]
 [0.26024294 0.14904018]
 [0.49753196 0.66502488]
 [0.63809597 0.23368568]
 [0.41058735 0.62320407]
 [0.68076054 0.46690813]
 [0.26930181 0.45360136]
 [0.45033375 0.02495853]
 [0.93062388 0.11089886]
 [0.61377448 0.34887483]
 [0.24330808 0.90028284]
 [0.47571335 0.3124448 ]
 [0.6941431  0.0668423 ]
 [0.37514126 0.88967978]]
Ycorrect is :
[[0], [0], [1], [1], [1], [1], [1], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [0], [1], [1], [0]]
w1 init is :
[[-0.8113182   1.4845988   0.06532937 -2.4427042   0.0992484 ]
 [ 0.5912243   0.59282297 -2.1229296  -0.72289723 -0.05627038]]
w2 init is :
[[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]
 [-2.4427042 ]
 [ 0.0992484 ]]


After 0 training step(s), loss_mse on all data is 22.8746
After 300 training step(s), loss_mse on all data is 1.09472
After 600 training step(s), loss_mse on all data is 0.765799
After 900 training step(s), loss_mse on all data is 0.634708
After 1200 training step(s), loss_mse on all data is 0.56278
After 1500 training step(s), loss_mse on all data is 0.519641
After 1800 training step(s), loss_mse on all data is 0.492133
After 2100 training step(s), loss_mse on all data is 0.473763
After 2400 training step(s), loss_mse on all data is 0.461043
After 2700 training step(s), loss_mse on all data is 0.451978


w1 trained is :
[[-0.19583124  0.6670919  -0.36684164 -1.051051    0.02691582]
 [ 0.68504494  0.40614486 -2.1118963  -0.41800624 -0.06789102]]
w2 trained is:
[[-0.39644957]
 [ 0.5086974 ]
 [ 0.3005259 ]
 [-0.8696754 ]
 [ 0.04641947]]

'''


