#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is an example implementation of ALS for learning how to use Spark. Please refer to
pyspark.ml.recommendation.ALS for more conventional use.

This example requires numpy (http://www.numpy.org/)
"""
from __future__ import print_function

import sys

import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession

LAMBDA = 0.01   # regularization
np.random.seed(42)


def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))

def update(i, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T # ratings[i, :].T 取ratings的第i行,注意要转置,变成U x 1

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu # 对角线上元素+LAMBDA * uu
 
    print(XtX + eye(shape(XtX)[1]) * LAMBDA * uu)
    
    return np.linalg.solve(XtX, Xty)  # Ax = B 求出x


if __name__ == "__main__":

    """
    Usage: als [M] [U] [F] [iterations] [partitions]"
    """

    print("""WARN: This is a naive implementation of ALS and is given as an
      example. Please use pyspark.ml.recommendation.ALS for more
      conventional use.""", file=sys.stderr)

    spark = SparkSession\
        .builder\
        .appName("PythonALS")\
        .getOrCreate()

    sc = spark.sparkContext

    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    U = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    partitions = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
          (M, U, F, ITERATIONS, partitions))

    R = matrix(rand(M, F)) * matrix(rand(U, F).T)
    ms = matrix(rand(M, F))
    us = matrix(rand(U, F))

    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)
    print("init...")
    print("Rb",Rb.value)
    print("msb",msb.value)
    print("usb",usb.value)

    for i in range(ITERATIONS):
	print("usb.value",usb.value)
	print("Rb.value",Rb.value)
        # 固定us,求ms （ALS）
	# sc.parallelize(range(M=5), partitions)会生成包含5个元素的列表，每一个元素包含了每一个用户的参数向量	
	# 取Rb.value的第i行，即求第i个电影的参数向量要用到第i行的评分
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, usb.value, Rb.value)) \
               .collect() 
        # ms返回的是一个list
	"""
           ms: [matrix([[ 0.63298562],
        	[ 0.63436862],
        	[ 0.13560026]]), matrix([[ 0.40486044],
        	[ 0.23138708],
        	[-0.06348073]]), matrix([[ 0.40901023],
        	[ 0.49780182],
        	[ 0.16234458]]), matrix([[ 0.58089794],
        	[ 0.43249159],
        	[ 0.01484741]]), matrix([[ 0.55480632],
        	[ 0.31027753],
        	[-0.09353143]])]
        """
        print("ms:::",ms)
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0]) # M x F matrix
        """
          ms:
	  e.g.
	    [[ 0.63298562  0.63436862  0.13560026]
 	    [ 0.40486044  0.23138708 -0.06348073]
 	    [ 0.40901023  0.49780182  0.16234458]
 	    [ 0.58089794  0.43249159  0.01484741]
 	    [ 0.55480632  0.31027753 -0.09353143]]
	"""
        # 放进广播变量
        msb = sc.broadcast(ms)
        # 固定ms，求us （ALS）
	# sc.parallelize(range(U=4), partitions)会生成包含4个元素的列表，每一个元素包含了每一个电影的参数向量
	# 取Rb.value的转置，Rb.T的第i行表示了第i个用户的评分，即求第i个用户的参数向量要用到第i行的评分	
        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, msb.value, Rb.value.T)) \
               .collect()
        us = matrix(np.array(us)[:, :, 0])
        # 放进广播变量
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % error)
        print("R",R)
	print("ms",ms)
        print("us",us)
	print("ms * us.T",ms * us.T)

    spark.stop()
