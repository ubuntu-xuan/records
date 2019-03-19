# coding:utf-8

'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    

# inA与inB假定都是列向量
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


# 协同过滤算法
# dataMat 用户数据 user 用户 simMeas 相似度计算方式 item 物品
# 计算未评分物品的估计评分时，是计算未评分物品与其它所有已评分物品的相似度与评分的累加
def standEst(dataMat, user, simMeas, item):
    print "对未评分物品求相似度：", item
    n = shape(dataMat)[1]  # 计算列的数量，物品的数量
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        print "对物品列：", j
        userRating = dataMat[user,j]
        print "userRating", userRating
        if userRating == 0:
          print "当前用户对该物品也没评分" 
          continue
        # numpy: np.logical_and/or/not (逻辑与/或/非) 
        print "当前用户对该物品已评分" 
        print "dataMat[:,item]", dataMat[:,item]
        # dataMat[:,item]： 取出item这一列数据

        # overLap取出对当前两个物品都有评分的行
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
	# nonzero返回的是不为0的元素的坐标nonzero[0]表示的是x轴，nonzero[1]表示的是y轴 
        print "overLap", overLap  # overLap是一个列表：如 [0,1,2,3,4]
 
        if len(overLap) == 0: similarity = 0

       # dataMat[overLap,item] 会取出所有overLap行item列的评分
        print "dataMat[overLap,item]", dataMat[overLap,item]
        print "dataMat[overLap,j]", dataMat[overLap,j]
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])

        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity #相似度累加
        print "simTotal", simTotal
        # 每次计算时还考虑相似度和当前用户评分的乘积
        ratSimTotal += similarity * userRating
        print  "ratSimTotal", ratSimTotal

    if simTotal == 0: return 0
    else:
        print  "ratSimTotal/simTotal", ratSimTotal/simTotal
        return ratSimTotal/simTotal
    
def svdEst(dataMat, user, simMeas, item):
    '''
        利用SVD分解先对dataMat降维
    '''
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    print "Sig4", Sig4
    # #根据k的值将原始数据转换到k维空间(低维),xformedItems表示物品(item)在k维空间转换后的值
    # xformedItems即Vnxk  变成n行k列
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        #xformedItems[item,:] 取出第item行
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=svdEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items 
    print "未评分的物品", unratedItems
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        print "对物品：", item
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    print "sorted:: ", sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
    # 返回前N个评分最高的未评价物品
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print 1,
            else: print 0,
        print ''

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print "****original matrix******"
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    print 'U numSV',shape(U[:,:numSV])
    print 'VT numSV', shape(VT[:numSV,:])
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print "****reconstructed matrix using %d singular values******" % numSV
    printMat(reconMat, thresh)

if __name__ == '__main__':
    #Data = loadExData()
    #U, Sigma, VT = linalg.svd(Data)
    myMat = mat(loadExData())
    myMat[0,1] = myMat[0,0] = myMat[1,0]  = myMat[2,0] = 4
    myMat[3,3] = 2 
    print myMat
    recommend(myMat, 2)
    #imgCompress()
