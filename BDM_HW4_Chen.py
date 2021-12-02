import csv
import pyspark
import json
import numpy as np
import sys
from datetime import datetime, timedelta


sc = pyspark.SparkContext()

outputFolder = sys.argv[1]
allStore = set(['452210','452311','445120','722410','722511','722513','446110','446191','311811', '722515','445210', '445220', '445230', '445291', '445292','445299','445110'])
storeLabel = {
	'452210': 0, '452311': 0, 
	'445120': 1, 
	'722410': 2, 
	'722511': 3, 
	'722513': 4, 
	'446110': 5, '446191': 5, 
	'722515': 6, '311811': 6, 
	'445210': 7, '445299': 7, '445230': 7, '445291': 7, '445220': 7, '445292': 7, 
	'445110': 8
}

corePlaces = sc.textFile('hdfs:///data/share/bdm/core-places-nyc.csv') \
				.filter(lambda x: next(csv.reader([x]))[9] in allStore)

allStoreSG = corePlaces.map(lambda x: next(csv.reader([x]))) \
              .map(lambda x : (x[1], storeLabel[x[9]]))

allStoreTotal = dict(allStoreSG.map(lambda x: (x[1],1)).reduceByKey(lambda x,y: x+y).collect())

allStoreLabel = dict(allStoreSG.collect())

def collectData(lines):
  for line in lines:
    line = next(csv.reader([line]))
    if line[1] in allStoreLabel:
      yield (allStoreLabel[line[1]], line[12], line[16])
'''
def getVisits(x):
  start = datetime.strptime(x[1][:10], '%Y-%m-%d')
  visits = json.loads(x[2])
  dayDiff = (start-datetime(2019,1,1)).days
  for i in range(7):
    date = datetime(2019,1,1)+timedelta(days=dayDiff)
    dayDiff+=1
    if date.year in [2019, 2020]:
      yield ((x[0], str(date.year), date.strftime("%m/%d/%Y, %H:%M:%S")[:6]+'2020'), [visits[i]])
'''
def getVisits(x):
  date = datetime.strptime(x[1][:10], '%Y-%m-%d')
  visits = json.loads(x[2])
  for i in visits:
    date += timedelta(days=1)
    if date.year in [2019, 2020]:
      yield ((x[0], str(date.year), date.strftime("%m/%d/%Y, %H:%M:%S")[:6]+'2020'), [i])


def lowMedianHigh(data):
  diff = allStoreTotal[data[0][0]] - len(data[1])
  std = int(np.std(data[1]+[0]*diff))
  med = int(np.std(data[1]+[0]*diff))
  low = max(0, med-std)
  high = max(0, med+std)
  return (data[0], med, low, high)


weeklyPattern = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*')
rdd = weeklyPattern.mapPartitions(collectData)\
							.flatMap(getVisits)\
							.reduceByKey(lambda a, b: a+b)\
							.map(lowMedianHigh)\
							.sortBy(lambda x: x[0][1]+x[0][2])


header = sc.parallelize([f'year,date,median,low,high'])

output = rdd.map(lambda x: (x[0][0], f'{x[0][1]},{x[0][2]},{x[1]},{x[2]},{x[3]}'))

outputFile = ['big_box_grocers',
           'convenience_stores',
           'drinking_places',
           'full_service_restaurants',
           'limited_service_restaurants',
           'pharmacies_and_drug_stores',
           'snack_and_bakeries',
           'specialty_food_stores',
           'supermarkets_except_convenience_stores']

for i, fileName in enumerate(outputFile):
  data = output.filter(lambda x: x[0] == i).map(lambda x: x[1]).coalesce(10)
  (header + data).saveAsTextFile(f'test/{fileName}')