from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import time

def Producer():
    producer = KafkaProducer(bootstrap_servers=["172.16.128.250:9092"])
    with open('/home/xuan/AI/data/ml-100k/u.user','r') as f:
	for line in f.readlines():
	    time.sleep(1)
	    producer.send("txt",line)
	    print line
	    #producer.flush()

if __name__ == '__main__':
    Producer()
