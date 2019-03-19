# -*- coding: UTF-8 -*-

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame,SQLContext,HiveContext
from pyspark import SparkFiles
import parse_csv as pycsv


sc = SparkContext()
# sqlCtx = SQLContext or HiveContext
#sqlCtx=SQLContext(sc)

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python load csv to DataFrame example") \
        .getOrCreate()

    # 将parse_csv.py上传到spark集群,否则spark-submit会提示找不到此文件
    sc.addPyFile('/home/xuan/mystudy/pyspark/csvToDataFrame/parse_csv.py')

    # Read csv data via SparkContext and convert it to DataFrame
    # load with rdd
    # 不带表头
    plaintext_rdd = sc.textFile(
        "hdfs:///ubuntuxuan/MyData/Titanic/train_without_header.csv")
    dataframe = pycsv.csvToDataFrame(spark, plaintext_rdd,columns=["PassengerId","Survived","Pclass","Name","Sex","Ag","SibSp","Parch","Ticket","Fare","Cabin","Embarked"],parseDate=False)
    dataframe.show()
    
    #带表头，自行推出字段类型
    plaintext_rdd = sc.textFile(
        "hdfs:///ubuntuxuan/MyData/Titanic/train_with_header.csv")
    dataframe = pycsv.csvToDataFrame(spark, plaintext_rdd,columns=None,parseDate=False)
    dataframe.show()

    spark.stop()

