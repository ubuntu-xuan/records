# https://www.cnblogs.com/tbiiann/p/6535189.html
# -- coding:utf-8 --

import numpy as np
import math

def lfm(a,k):
    '''
    参数a：表示需要分解的评价矩阵
    参数k：分解的属性（隐变量）个数
    '''
    assert type(a) == np.ndarray
    m, n = a.shape
    alpha = 0.01
    lambda_ = 0.01
    u = np.random.rand(m,k)
    v = np.random.randn(k,n)

    print "初始化"
    print 'u',u
    print 'v',v
    
    '''
        使用了随机梯度下降法SGD，不是交替最小二乘ALS
    '''

    for t in range(2): # 迭代次数
	print "*********************迭代:***********************************",t+1
        for i in range(m):
	    print "-----------------电影i:--------------------------------",i+1
            for j in range(n):
		print "+++++++++++++用户j:++++++++++++++",j+1
                if math.fabs(a[i][j]) > 1e-4: #评分不为0
		    print "math.fabs(a[i][j]) > 1e-4",math.fabs(a[i][j]) 		    
                    err = a[i][j] - np.dot(u[i],v[:,j])
                    for r in range(k):
			print "v[r][j]",v[r][j]
			print "u[i][r]",u[i][r]
 			print "求U的参数向量"
                        gu = err * v[r][j] - lambda_ * u[i][r]
			print "求V的参数向量"
                        gv = err * u[i][r] - lambda_ * v[r][j]
                        print "更新U的参数向量"
                        u[i][r] += alpha * gu
			print 'u[ir]:',u[i][r]	
			print 'U对应变化位置:'
			print "电影i:",i+1
			print "隐变量k:",r+1
			print 'u',u

			print "更新V的参数向量"
                        v[r][j] += alpha * gv
		 	print 'v[rj]:',v[r][j]
	                print 'V对应变化位置:'
                        print "用户j:",j+1
                        print "隐变量k:",r+1
		        print 'v',v

        print "迭代一次完成:",u,v
    return u,v 

A = np.array([[5,5,0,5],[5,0,3,4],[3,4,0,3],[0,0,5,3],[5,4,4,5],[5,4,5,5]])
b,c = lfm(A,3)

print 'A',A


