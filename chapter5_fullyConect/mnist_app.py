#coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward

def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y, 1)		#预测结果

		#实例化带滑动平均值的saver
		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
 		variables_to_restore = variable_averages.variables_to_restore()
 		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)		#加载模型参数
		
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("没找到模型！")
				return -1
#图片预处理
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS)		#消除锯齿方法打开图片，调整大小
	im_arr = np.array(reIm.convert('L'))
	threshold = 50		#图像降噪阈值
	#转色处理  因为输入的图片和训练的图片颜色正好相反
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
 			if (im_arr[i][j] < threshold):		
 				im_arr[i][j] = 0		#小于指定阈值令为纯黑色
			else: im_arr[i][j] = 255

	nm_arr = im_arr.reshape([1, 784])		#调整为训练数组维度
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)		#归一化

	return img_ready

def application():
	testNum = input("输入测试图片数量： ")
	for i in range(testNum):
		testPic = raw_input("输入当前测试图片完整路径：")
		testPicArr = pre_pic(testPic)
		preValue = restore_model(testPicArr)
		print "预测该数字为:", preValue
		testflag = raw_input("输入决定是否继续测试?:")
		if testflag is "n":
			break
		else:
			continue
	print "测试结束！"

def main():
	application()

if __name__ == '__main__':
	main()		
