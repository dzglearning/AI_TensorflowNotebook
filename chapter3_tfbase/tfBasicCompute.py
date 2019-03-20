#coding:utf-8
#两层全连接神经网络

import tensorflow as tf


#用placeholder定义输入（sess.run喂一组或多组数据）
#x = tf.placeholder(tf.float32, shape=(1, 2))		#一组，两个属性列
x = tf.placeholder(tf.float32, shape=(None, 2))		#多组，用None占位
w1= tf.Variable(tf.random_normal([2, 5], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([5, 1], stddev=1, seed=1))		#隐含层有五个节点

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#调用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  
    sess.run(init_op)		#初始化数据
    #print"test y is:\n",sess.run(y, feed_dict={x: [[0.7,0.5]]})
    print "the result of test is:\n",sess.run(y, feed_dict={x: [[0.7,0.5],[0.2,0.3],[0.4,0.5]]})	#喂入多组数据，输出y
    print "w1 is :\n", sess.run(w1)
    print "w2 is :\n", sess.run(w2)		#节点参数输出

'''
test y is:单组结果
[[7.201226]]

the result of test is: 多组结果
[[7.201226 ]
 [2.3752337]
 [4.5482683]]

随机种子相同 随机数也相同
w1 is :
[[-0.8113182   1.4845988   0.06532937 -2.4427042   0.0992484 ]
 [ 0.5912243   0.59282297 -2.1229296  -0.72289723 -0.05627038]]
w2 is :
[[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]
 [-2.4427042 ]
 [ 0.0992484 ]]

'''
