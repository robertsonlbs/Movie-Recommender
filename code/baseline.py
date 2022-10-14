#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true baseline.py <small/full>

'''
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics
import getpass
import sys
import builtins as py_builtin


def main(spark, netID, data_size):
    '''
    Main routine for Data preprocessing and partitions
    Parameters
    ----------
    spark : SparkSession object
    '''
    k = 100
    if data_size == 'small':
        train_path = f'hdfs:/user/{netID}/small/train_df.csv'
        val_path = f'hdfs:/user/{netID}/small/val_df.csv'
        test_path = f'hdfs:/user/{netID}/small/test_df.csv'
    if data_size == 'full':
        train_path = f'hdfs:/user/{netID}/full/train_df.csv'
        val_path = f'hdfs:/user/{netID}/full/val_df.csv'
        test_path = f'hdfs:/user/{netID}/full/test_df.csv'

    # schema (userId=148, movieId=44191, rating=4.0, timestamp=1482550089, row_number=1)
    train_df = spark.read.csv(train_path, header=True, quote='"', sep=",", inferSchema=True)
    val_df = spark.read.csv(val_path, header=True, quote='"', sep=",", inferSchema=True)
    test_df = spark.read.csv(test_path, header=True, quote='"', sep=",", inferSchema=True)
    # return the top 100 recommendations
    prior_lst = [1, 5, 10, 20, 50, 100]
    map_lst = []
    for prior in prior_lst:
        recom = fit(train_df, prior)
        map_val, precision_val, ndcg_val = report(val_df, recom, k)
        map_lst.append((map_val, prior))
        print(f"Validation Set: prior={prior}, Map={map_val}")

    best_prior = py_builtin.max(map_lst)[1]
    print(f"Validation Set Best Performance: At prior= {best_prior}")
    recom = fit(train_df, best_prior)
    map_test, precision_test, ndcg_test = report(test_df, recom, k)
    print(f"Test Set Final Performance: Map={map_test}, Precision={precision_test}, NDCG={ndcg_test}")


def report(evaluation_df, recom, k):
    recom_item = recom.agg(collect_list('movieId').alias('recom_item'))
    true_recom_combined = evaluation_df.groupby('userId').agg(collect_list('movieId').alias('true_item')).join(
        recom_item)
    predictionAndLabels = true_recom_combined.rdd.map(lambda row: (row[2], row[1]))
    metrics = RankingMetrics(predictionAndLabels)
    map_ = metrics.meanAveragePrecision
    precision = metrics.precisionAt(k)
    ndcg = metrics.ndcgAt(k)
    return map_, precision, ndcg


def fit(train_df, prior):
    '''
    Return A spark Dataframe of the movie recommendations(top 100)
    ----------
    ('movieId','ajusted')
    
    Parameters
    ----------
    train_df: Spark Dataframe for training
    prior: penalty for 

    '''
    base = train_df.groupby('movieId').agg(count('rating').alias('count'), sum('rating').alias('total_score'))
    recom = base.select(col('movieId'), (col('total_score') / (col('count') + prior)).alias('adjusted')).sort(
        col('adjusted'), ascending=False).limit(100)
    return recom


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('Recomm_Sys_Data').getOrCreate()
    netID = getpass.getuser()
    data_size = sys.argv[1]
    # Call our main routine
    main(spark, netID, data_size)
