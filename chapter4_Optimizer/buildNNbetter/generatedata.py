#coding:utf-8
#生成数据集

import numpy as np
import matplotlib.pyplot as plt
seed = 12306 
def generateds():
	rdm = np.random.RandomState(seed)
	X = rdm.randn(300,2)
	Y_ = [int(x0*x0 + x1*x1 <2) for (x0,x1) in X]
	Y_co = [['red' if y else 'blue'] for y in Y_]
	X = np.vstack(X).reshape(-1,2)
	Y_ = np.vstack(Y_).reshape(-1,1)
	print"DATA has generated!!\n"

	return X, Y_, Y_co
