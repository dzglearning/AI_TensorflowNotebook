#coding:utf-8
#对比正则化和没有正则化训练的网络模型预测区别

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batchsize = 30 
seed = 12306 
lr_rate = 0.0001
regular = 0.01		#正则化权重参数为 0.01

rdm = np.random.RandomState(seed)
X = rdm.randn(300,2)
#300 组随机数组成的坐标

#两个坐标的平方和小于2，给Y赋值1，其余赋值0  作为输入数据集的标签
Y_ = [int(x0*x0 + x1*x1 <2) for (x0,x1) in X]

#着色  1赋值red 其余赋值blue 
Y_color = [['red' if y else 'blue'] for y in Y_]

#变形，第一个元素-1指变形后形状随第二个参数计算获得，把X整理为n行2列，把Y整理为n行1列
X = np.vstack(X).reshape(-1,2)
Y_ = np.vstack(Y_).reshape(-1,1)

print X
print Y_
print Y_color       #输出自制数据集
print"DATA has generated!!\n"

#画出数据集X中每组（x0，x1），用Y_color对应的值表示颜色
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_color)) 
plt.title("DATA distributions")
plt.show()


#前向传播过程 
def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)      #按shape生成参数矩阵
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))        #L2正则
	print"Weights has got!! \n"	

	return w
#取权重参数
     
def get_bias(shape):  
    b = tf.Variable(tf.constant(0.01, shape=shape))
    print"basis has got!! \n"
 
    return b
#取偏置
	
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2,16], regular)	#两层隐含层 16 个节点,正则化权重参数regular
b1 = get_bias([16])
y1 = tf.nn.relu(tf.matmul(x, w1)+b1)    #激活函数relu

w2 = get_weight([16,1], regular)       #第二层
b2 = get_bias([1])
y = tf.matmul(y1, w2)+b2            #输出不激活


#定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))       #带正则loss


#反向传播：不含正则化
train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_mse)
#train_step = tf.train.GradientDescentOptimizer(lr_rate).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(lr_rate,0.9).minimize(loss_mse)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 60000
	for i in range(STEPS):
		start = (i*batchsize) % 300 #取数据都落在区间内
		end = start + batchsize
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
		if i % 2000 == 0:
			loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
			print("经过 %d 轮, loss 值为: %f" %(i, loss_mse_v))
            
    #生成二维网格坐标点，选一个区域
	xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
	#拉直， 按行 合并成一个2列矩阵，得到坐标集合
	grid = np.c_[xx.ravel(), yy.ravel()] 
	#将网格坐标点喂入神经网络 ，probs为输出，表示一种倾向性，倾向蓝色还是红色
	probs = sess.run(y, feed_dict={x:grid})
	#probs的shape调整成xx的样子
	probs = probs.reshape(xx.shape)
	print "未正则化反向传播参数w1:\n",sess.run(w1)
	print "未正则化反向传播参数b1:\n",sess.run(b1)
	print "未正则化反向传播参数w2:\n",sess.run(w2)	
	print "未正则化反向传播参数b2:\n",sess.run(b2)

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_color))
plt.contour(xx, yy, probs, levels=[.5])     
#等高线绘制， probs表示蓝红倾向性深浅(高度)，level决定绘制出来显示的高度，位置
plt.title("Build NN Examples without Regularizer")
plt.show()



#定义反向传播方法：包含正则化
train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_total)
#train_step = tf.train.GradientDescentOptimizer(lr_rate).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(lr_rate,0.9).minimize(loss_mse)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 60000
	for i in range(STEPS):
		start = (i*batchsize) % 300
		end = start + batchsize
		sess.run(train_step, feed_dict={x: X[start:end], y_:Y_[start:end]})
		if i % 2000 == 0:
			loss_v = sess.run(loss_total, feed_dict={x:X,y_:Y_})
			print("经过 %d steps, loss 值为: %f" %(i, loss_v))

	xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	probs = sess.run(y, feed_dict={x:grid})
	probs = probs.reshape(xx.shape)
	print "经过正则化反向传播参数w1:\n",sess.run(w1)
	print "经过正则化反向传播参数b1:\n",sess.run(b1)
	print "经过正则化反向传播参数w2:\n",sess.run(w2)
	print "经过正则化反向传播参数b2:\n",sess.run(b2)

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_color)) 
plt.contour(xx, yy, probs, levels=[.5])
plt.title("Build NN Examples with Regularizer")
plt.show()

