# -*- coding:utf-8 -*-

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

#import sys
#reload(sys)
#sys.setdefaultcoding('utf-8')


TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:  #复现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)      #前向传播计算输出y

        #实例化带滑动平均的saver对象
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        #所有参数在会话中被加载时会被赋值为各自的滑动平均值
		
        #计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                #加载模型，赋参数滑动平均值
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path) #恢复到当前会话
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print "after %s training steps(s), test accuracy = %g"%(global_step, accuracy_score)
                else:
                    print 'no found model'
                    return
            time.sleep(TEST_INTERVAL_SECS)
	    flags = raw_input("输入y或n决定是否继续：")
	    if flags == "y":
		continue
	    else:
		break

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()
