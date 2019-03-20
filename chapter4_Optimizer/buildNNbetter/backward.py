#coding:utf-8
#反向传播

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import generatedata
import forward
#导入自定义模块

STEPS = 60000
batchsize = 30 
lr_base = 0.001
lr_decay = 0.999
regular = 0.01

def backward():
	x = tf.placeholder(tf.float32, shape=(None, 2))
	y_ = tf.placeholder(tf.float32, shape=(None, 1))

	X, Y_, Y_c = generatedata.generateds()
	y = forward.forward(x, regular)
	
	global_step = tf.Variable(0,trainable=False)	

	learning_rate = tf.train.exponential_decay(
		lr_base,
		global_step,
		300/batchsize,
		lr_decay,
		staircase=True)


	#定义损失函数
	loss_mse = tf.reduce_mean(tf.square(y-y_))
	loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
	
	#定义反向传播方法：包含正则化
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
	#train_step = tf.train.GradientDescentOptimizer(learning_rat).minimize(loss_total)
	#train_step = tf.train.MomentumOptimizer(learning_rat,0.9).minimize(loss_total)

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		for i in range(STEPS):
			start = (i*batchsize) % 300
			end = start + batchsize
			sess.run(train_step, feed_dict={x: X[start:end], y_:Y_[start:end]})
			if i % 2000 == 0:
				loss_v = sess.run(loss_total, feed_dict={x:X,y_:Y_})
				print("经过 %d 轮训练, loss 值为: %f" %(i, loss_v))

		xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = sess.run(y, feed_dict={x:grid})
		probs = probs.reshape(xx.shape)
	
	plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c)) 
	plt.contour(xx, yy, probs, levels=[.5])
	plt.title("Build Neural Networks Examples")
	plt.show()
	
if __name__=='__main__':
	backward()
