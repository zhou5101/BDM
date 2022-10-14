from pyspark import SparkContext
import datetime
import csv
import functools
import json
import numpy as np
import sys

def main(sc):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
    OUTPUT_PREFIX = sys.argv[1]

    CAT_CODES = {'445210', '445110', '722410', '452311', '722513', '445120', '446110', '445299', '722515', '311811', '722511', '445230', '446191', '445291', '445220', '452210', '445292'}
    CAT_GROUP = {'452210': 0, '452311': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446110': 5, '446191': 5, '722515': 6, '311811': 6, '445210': 7, '445299': 7, '445230': 7, '445291': 7, '445220': 7, '445292': 7, '445110': 8}
    
    def filterPOIs(_, lines):
        for line in csv.reader(lines):
            if line[9] in CAT_CODES:
                yield (line[0], CAT_GROUP[line[9]])
    
    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs) \
            .cache()

    storeGroup = dict(rddD.collect())
    groupCount = rddD.map(lambda x: (x[1], 1))\
                    .reduceByKey(lambda x,y:x+y)\
                    .sortByKey()\
                    .values()\
                    .collect()

    def extractVisits(storeGroup, _, lines):
        for line in csv.reader(lines):
            if line[0] in storeGroup:
                visits = json.loads(line[16])
                date = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')
                startDate = datetime.datetime.strptime(line[12][:10], '%Y-%m-%d')
                dateRange = [startDate+datetime.timedelta(days=i) for i in range(7)]
                diff = [(i-date).days for i in dateRange]
                for i,j in zip(diff, visits):
                    if i>-1:
                        yield ((storeGroup[line[0]], i), j)

    rddG = rddPattern \
        .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))


    def computeStats(groupCount, _, records):
        for record in records:
            data = list(record[1])
            diff = groupCount[record[0][0]]-len(data)
            std = np.std(diff*[0]+data)
            med = np.median(diff*[0]+data)
            low = min(0, std-med)
            high = std+med
            yield (record[0],(med, low, high))

    def formatData(_, records):
        for record in records:
            date = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')+ datetime.timedelta(days=record[0][1])
            formatted = date.strftime('%m-%d-%Y')[:5]
            yield (record[0][0],f'{date.year},2020-{formatted},{record[1][0]},{record[1][1]},{record[1][2]}')

    rddH = rddG.groupByKey() \
        .mapPartitionsWithIndex(functools.partial(computeStats, groupCount))

    rddI = rddH.mapPartitionsWithIndex(functools.partial(formatData))

    rddJ = rddI.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    rddJ = (header + rddJ).coalesce(10).cache()

    outputs = ['big_box_grocers',
    'convenience_stores',
    'drinking_places',
    'full_service_restaurants',
    'limited_service_restaurants',
    'pharmacies_and_drug_stores',
    'snack_and_retail_bakeries',
    'specialty_food_stores',
    'supermarkets_except_convenience_stores']

    for i, filename in enumerate(outputs):
        rddJ.filter(lambda x: x[0]==i or x[0]==-1).values() \
            .saveAsTextFile(f'{OUTPUT_PREFIX}/{filename}')
if __name__=='__main__':
    sc = SparkContext()
    main(sc)