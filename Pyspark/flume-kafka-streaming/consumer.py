https://blog.csdn.net/chenyulancn/article/details/79420522

# -*- coding:utf-8 -*-

from pyspark import SparkContext  
from pyspark import SparkConf  
from pyspark.streaming import StreamingContext  
from pyspark.streaming.kafka import KafkaUtils,TopicAndPartition  
import MySQLdb  

def start():
    sconf = SparkConf()
    sc=SparkContext(appName='txt',conf=sconf) 
    ssc=StreamingContext(sc,5)
    brokers = "172.16.128.201:9092,172.16.128.250:9092"
    topic='txt'  
    start = 70000  
    partition=0  
    user_data = KafkaUtils.createDirectStream(ssc,[topic],kafkaParams={"metadata.broker.list":brokers})  
    #user_data = KafkaUtils.createDirectStream(ssc,[topic],kafkaParams={"metadata.broker.list":brokers},fromOffsets={TopicAndPartition(topic,partition):long(start)})  
    #user_data.pprint()
    user_fields = user_data.map(lambda line: line[1].split('|'))  
    gender_users = user_fields.map(lambda fields: fields[2]).map(lambda gender: (gender,1)).reduceByKey(lambda a,b: a+b)  

    user_data.foreachRDD(offset) #存储offset信息  
    gender_users.foreachRDD(lambda rdd: rdd.foreach(echo))

    ssc.start()  
    ssc.awaitTermination()  

offsetRanges = []  
def offset(rdd):  
    # rdd: [(None, u'843|35|M|librarian|44212\n'), (None, u'844|22|M|engineer|95662\n'), (None, u'845|64|M|doctor|97405\n'), (None, u'846|27|M|lawyer|47130\n')]
    global offsetRanges  
    offsetRanges = rdd.offsetRanges()  
    for o in offsetRanges:
        topic = o.topic
        partition = o.partition
        fromoffset = o.fromOffset
        untiloffset = o.untilOffset

def echo(rdd):  
    zhiye = rdd[0]  
    num = rdd[1]  
    print "offsetRanges",offsetRanges
    for o in offsetRanges:  
        topic = o.topic   
        partition = o.partition  
        fromoffset = o.fromOffset  
        untiloffset = o.untilOffset  

    # need create tabel in DataNode(172.16.128.201)
    conn = MySQLdb.connect('localhost','root','uroot012','test',charset='utf8')
    cursor = conn.cursor()  
    sql = "insert into zhiye(zhiye,num) \
                       values ('%s','%d')" % (zhiye,num)  
    cursor.execute(sql)  
    conn.commit()  
    conn.close()   


if __name__ == '__main__':  
    start()  
