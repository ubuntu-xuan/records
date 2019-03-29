# -*-coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)


def loadDataSet(fileName):
	"""
	函数说明:加载数据
	Parameters:
		fileName - 文件名
	Returns:
		xArr - x数据集
		yArr - y数据集
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	numFeat = len(open(fileName).readline().split('\t')) - 1
	xArr = []; yArr = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		xArr.append(lineArr)
		yArr.append(float(curLine[-1]))
	return xArr, yArr

def ridgeRegres(xMat, yMat, lam = 0.2):
	"""
	函数说明:岭回归
  	  岭回归： 即是加入了L2正则化的最小二乘
  	        w = (X.T * X  + lambda * E).I * X.T * y （y是列向量）
	Parameters:
		xMat - x数据集
		yMat - y数据集
		lam - 缩减系数
	Returns:
		ws - 回归系数
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam   # 岭回归多加了 lam*I
	if np.linalg.det(denom) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = denom.I * (xMat.T * yMat)   #   .I  求逆
	print('ws--------------', ws)
	return ws

def ridgeTest(xArr, yArr):
	"""
	函数说明:岭回归测试
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		wMat - 回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	#数据标准化
	yMean = np.mean(yMat, axis = 0)					#行与行操作，求均值
	yMat = yMat - yMean							#数据减去均值

	xMeans = np.mean(xMat, axis = 0)					#行与行操作，求均值
	xVar = np.var(xMat, axis = 0)						#行与行操作，求方差
	xMat = (xMat - xMeans) / xVar						#数据减去均值除以方差实现标准化
        # wMat存储了30个在不同lambda中计算到的w
	numTestPts = 30						#30个不同的lambda测试
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))	#初始回归系数矩阵
	for i in range(numTestPts):				#改变lambda计算回归系数
		ws = ridgeRegres(xMat, yMat, np.exp(i - 10))	#lambda以e的指数变化，最初是一个非常小的数，
		print('np.exp(i - 10)', np.exp(i - 10))
		wMat[i, :] = ws.T 				#计算回归系数矩阵
	return wMat

def plotwMat():
	"""
	函数说明:绘制岭回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	font = FontProperties(size=14)
	abX, abY = loadDataSet('abalone.txt')
	redgeWeights = ridgeTest(abX, abY)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# redgeWeights: 在30个不同lam下，求得30组回归系数

	print('redgeWeights-----------------', redgeWeights)
	# print('redgeWeights[0][0]', redgeWeights[0])
	# data = []
	# for i in redgeWeights:
	#     print('i',i[0])
	#     data.append(i[0])

	#lambdas = [i-10 for i  in range(30)]
	#ax.plot(lambdas, redgeWeights)


	ax.plot(redgeWeights)	
	ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系')
	ax_xlabel_text = ax.set_xlabel(u'log(lambada)')
	ax_ylabel_text = ax.set_ylabel(u'回归系数')
	plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()


def regularize(xMat, yMat):
	"""
	函数说明:数据标准化
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		inxMat - 标准化后的x数据集
		inyMat - 标准化后的y数据集
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""	
	inxMat = xMat.copy()														#数据拷贝
	inyMat = yMat.copy()
	yMean = np.mean(yMat, 0)													#行与行操作，求均值
	inyMat = yMat - yMean														#数据减去均值
	inMeans = np.mean(inxMat, 0)   												#行与行操作，求均值
	inVar = np.var(inxMat, 0)     												#行与行操作，求方差
	inxMat = (inxMat - inMeans) / inVar											#数据减去均值除以方差实现标准化
	return inxMat, inyMat

def rssError(yArr,yHatArr):
	"""
	函数说明:计算平方误差
	Parameters:
		yArr - 预测值
		yHatArr - 真实值
	Returns:
		
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""
	return ((yArr-yHatArr)**2).sum()

def stageWise(xArr, yArr, eps = 0.01, numIt = 2):
	"""
	函数说明:前向逐步线性回归,可以得到与Lasso回归差不多的效果，但更简单
        Lasso回归：加入了L1正则化的最小二乘
	Parameters:
		xArr - x输入数据
		yArr - y预测数据
		eps - 每次迭代需要调整的步长
		numIt - 迭代次数
	Returns:
		returnMat - numIt次迭代的回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""

	'''
	    属于一种贪心算法，给定初始系数向量，然后不断迭代遍历每个系数，增加或减小一个很小的数，
	    看看代价函数是否变小，如果变小就保留，如果变大就舍弃，然后不断迭代直到回归系数达到稳定
	    ps:
	        每一次迭代只会改变一个特征（使平方误差最小的那个）
	'''


	xMat = np.mat(xArr); yMat = np.mat(yArr).T 		#数据集
	xMat, yMat = regularize(xMat, yMat)			#数据标准化
	m, n = np.shape(xMat)					#  m：样本数目  n：特征数目
	returnMat = np.zeros((numIt, n))			#初始化numIt次迭代的回归系数矩阵
	ws = np.zeros((n, 1))					#初始化回归系数矩阵
	# print("初始化回归系数矩阵")
	# print(ws)

	wsTest = ws.copy()
	wsMax = ws.copy()

	for i in range(numIt):					#迭代numIt次
		# print("第%s次迭代"%(i+1))
		'''
		    第n次迭代会保留第n-1次迭代的ws
		'''
		# print(ws.T)					#打印当前回归系数矩阵
		lowestError = float('inf'); 			#正无穷
		for j in range(n):				#遍历每个特征的回归系数
			#print("对第%s个特征"%j)
			for sign in [-1, 1]:
				#print('sign:',sign)
				wsTest = ws.copy()	# copy出来的对象会独立出来，修改本体时不受影响	
				#print('-------wsTest-------', wsTest)
				wsTest[j] += eps * sign		#微调回归系数
				#print('++++++wsTest++++++', wsTest)
				yTest = xMat * wsTest		#计算预测值
				rssE = rssError(yMat.A, yTest.A)		#计算平方误差
				if rssE < lowestError:			#如果误差更小，则更新当前的最佳回归系数
					#print('误差为:%s', rssE)
					#print('误差更小更新当前的最佳回归系数')
					lowestError = rssE
					wsMax = wsTest
					#print('将微调后的回归系数赋值给wsMax')
					#print(wsMax)
				#else:
				     #print("不需要经过微调，此时的waMax:")
				     #print(wsMax)

				'''
				    当sign为1和-1时，都不经过微调，这时候对应的特征的w为0
				'''

			#print("对第%s个特征得到wsMax"%j)
			#print('wsMax', wsMax)

		'''
		    每一次迭代只会改变一个特征（使平方误差最小的那个）		    
		'''
		#print('遍历完所有特征,保存本次迭代经过微调得到的回归系数')
		#print(wsMax.copy())

		ws = wsMax.copy()

		#print('记录第%s次迭代的回归系数矩阵'%i)
		returnMat[i,:] = ws.T 					#记录第numIt次迭代的回归系数矩阵
		#print('returnMat', returnMat)
	return returnMat

def plotstageWiseMat():
	"""
	函数说明:绘制岭回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""
	font = FontProperties(size=14)
	xArr, yArr = loadDataSet('abalone.txt')
	returnMat = stageWise(xArr, yArr, 0.001, 5000)
	print('------------------------------------------------------+++++++++++++++++++')
	print(returnMat)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(returnMat)	

	ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系')
	ax_xlabel_text = ax.set_xlabel(u'迭代次数')
	ax_ylabel_text = ax.set_ylabel(u'回归系数')
	plt.setp(ax_title_text, size = 15, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()


'''
  交叉验证
'''
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)




if __name__ == '__main__':
	#plotwMat()
	plotstageWiseMat()
