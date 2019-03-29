# -- coding:utf-8 --

import numpy as np
from numpy.random import rand
from numpy import matrix

i = [0]
R = matrix(rand(5, 3)) * matrix(rand(4, 3).T)
ms = matrix(rand(5, 3))
us = matrix(rand(4, 3))
LAMBDA = 0.01

print("R",R)
print("us",us)


def update(i, mat, ratings):
    uu = mat.shape[0] #usb的行数
    print("usb的行数",uu)
    ff = mat.shape[1] #usb的列数
    print("#usb的列数",ff)

    XtX = mat.T * mat
    print("XtX",XtX) # print()与print 矩阵输出的格式不同，print()会以逗号分隔
    print("ratings[i, :].T",ratings[i, :].T)
    Xty = mat.T * ratings[i, :].T # ratings[i, :].T 取ratings的第i行,注意要转置

    for j in range(ff):
        print 'j',j
        print 'LAMBDA * uu',LAMBDA * uu

