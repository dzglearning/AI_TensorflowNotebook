#coding:utf-8
#损失函数优化参数w的过程

import tensorflow as tf

lr_base = 0.1   #最初学习率
lr_decay = 0.99     #学习率衰减率
lr_step = 1     #喂入多少轮batchsize后，更新（降低）一次学习率，一般取：总样本数/batchsize
global_step = tf.Variable(0, trainable=False)   #计数运算律几轮batchsize，初值0, 不作为被训练数

#指数衰减学习率，在迭代初期得到较高的下降速度，较小的训练轮数下取得更好收敛度
#lr = lr_base * lr_decay^(global_step/lr_step)
lr = tf.train.exponential_decay(lr_base, global_step, lr_step, lr_decay, staircase=True)    #True 梯形衰减，false平滑衰减

#待优化参数w，初值给10
w = tf.Variable(tf.constant(10, dtype=tf.float32))

#损失函数loss=(w+1)^2
loss = tf.square(w+1)

#反向传播训练，不声明global_step=global_step 的话  global_step值不变，学习率也不会变
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

#运算会话，训练50轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(50):
        sess.run(train_step)
        lr_val = sess.run(lr)   #动态更新学习率的数值
        global_step_val = sess.run(global_step)     #global_step通过反向传播处自加1
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print "经过 %s 轮训练: global_step 值为 %f, w 值为 %f, learning rate 值为 %f, loss 值为 %f" % (i, global_step_val, w_val, lr_val, loss_val)



'''
经过 0 轮训练: global_step 值为 1.000000, w 值为 7.800000, learning rate 值为 0.099000, loss 值为 77.440002
经过 1 轮训练: global_step 值为 2.000000, w 值为 6.057600, learning rate 值为 0.098010, loss 值为 49.809719
经过 2 轮训练: global_step 值为 3.000000, w 值为 4.674169, learning rate 值为 0.097030, loss 值为 32.196194
...
经过 48 轮训练: global_step 值为 49.000000, w 值为 -0.997732, learning rate 值为 0.061112, loss 值为 0.000005
经过 49 轮训练: global_step 值为 50.000000, w 值为 -0.998009, learning rate 值为 0.060501, loss 值为 0.000004

'''
