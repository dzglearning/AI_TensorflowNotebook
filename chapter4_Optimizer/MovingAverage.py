#coding:utf-8
#计算参数的滑动平均值（MOVING_AVERAGE）

import tensorflow as tf

#定义一个浮点变量，初值0.0  不断更新w1优化w1，滑动平均取w1过往影子值
w1 = tf.Variable(0, dtype=tf.float32)

#迭代轮数，参数不可被训练
global_step = tf.Variable(0, trainable=False)

#实例化滑动平均类，给衰减率为0.99，当前轮数global_step
mv_decay = 0.99
ema = tf.train.ExponentialMovingAverage(mv_decay, global_step)
ema_op = ema.apply(tf.trainable_variables())    #所有待优化参数列表求华东平均


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print "当前global_step值为：", sess.run(global_step)
    print "当前参数 w1 为：", sess.run([w1, ema.average(w1)]), '\n'    
    #打印出当前参数w1和w1滑动平均值      ema.average(w1)查看当前w1滑动平均值
    
    sess.run(tf.assign(w1, 1))    # 参数w1的值更改为1
    sess.run(ema_op)
    print "当前 global_step:", sess.run(global_step)
    print "当前 w1", sess.run([w1, ema.average(w1)]), '\n' 
    
    # 更新global_step为100，参数w1为10, 下面的global_step都为100，每次执行滑动平均操作，影子值会更新 
    sess.run(tf.assign(global_step, 100))  
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print "当前 global_step:", sess.run(global_step)
    print "当前 w1:", sess.run([w1, ema.average(w1)]), '\n'       
    
    # 每次sess.run会更新一次w1的滑动平均值
    sess.run(ema_op)
    print "当前 global_step:" , sess.run(global_step)
    print "当前 w1:", sess.run([w1, ema.average(w1)]), '\n' 

    sess.run(ema_op)
    print "当前 global_step:" , sess.run(global_step)
    print "当前 w1:", sess.run([w1, ema.average(w1)]), '\n' 

    sess.run(ema_op)
    print "当前 global_step:" , sess.run(global_step)
    print "当前 w1:", sess.run([w1, ema.average(w1)]), '\n' 

    sess.run(ema_op)
    print "当前 global_step:" , sess.run(global_step)
    print "当前 w1:", sess.run([w1, ema.average(w1)]), '\n' 

    sess.run(ema_op)
    print "当前 global_step:" , sess.run(global_step)
    print "当前 w1:", sess.run([w1, ema.average(w1)]), '\n' 

    sess.run(ema_op)
    print "当前 global_step:" , sess.run(global_step)
    print "当前 w1:", sess.run([w1, ema.average(w1)]), '\n' 


'''
当前global_step值为： 0
当前参数 w1 为： [0.0, 0.0] 

当前 global_step: 0
当前 w1 [1.0, 0.9] 

当前 global_step: 100
当前 w1: [10.0, 1.6445453] 

当前 global_step: 100
当前 w1: [10.0, 2.3281732] 

当前 global_step: 100
当前 w1: [10.0, 2.955868] 

当前 global_step: 100
当前 w1: [10.0, 3.532206] 

当前 global_step: 100
当前 w1: [10.0, 4.061389] 

当前 global_step: 100
当前 w1: [10.0, 4.547275] 

当前 global_step: 100
当前 w1: [10.0, 4.9934072] 

'''
