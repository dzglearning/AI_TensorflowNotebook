#coding:utf-8
import tensorflow as tf

INPUT_NODE = 784
# 神经网络输入节点，图片像素值784个点，一维数组
OUTPUT_NODE = 10
# 输出十个数，每个数索引号对应数字出现的概率
LAYER1_NODE = 500
# 隐藏层的及诶单个数

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    print "Got weights!\n"
    return w


def get_bias(shape):  
    b = tf.Variable(tf.zeros(shape))
    print "Got basis!\n"  
    return b
	
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # 第一层参数，偏置，输出

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    print "Completed forward compute!\n"
    # 要对输出使用softmax 函数，转化为概率分布，所以不过relu函数
    return y
