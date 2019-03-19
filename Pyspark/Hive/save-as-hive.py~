rom pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Saving to Persistent Tables .saveAsTable") \
        .enableHiveSupport() \
        .getOrCreate()
    # 查看hive中的表
    print("------"*20,spark.catalog.listTables())
    df = spark.read.csv("hdfs:///ubuntuxuan/MyData/Titanic/train_with_header.csv")
    #df.show()
    df.write.saveAsTable("testing")

