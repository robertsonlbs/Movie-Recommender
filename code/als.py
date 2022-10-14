#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true als.py <small/full>

"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
import getpass
import sys
import itertools
import builtins as py_builtin
import pandas as pd

def main(spark, netID, data_size):
    """
    Main routine for Data preprocessing and partitions
    Parameters
    ----------
    spark : SparkSession object
    """
    k = 100
    train_path = f'hdfs:/user/{netID}/{data_size}/train_df.csv'
    val_path = f'hdfs:/user/{netID}/{data_size}/val_df.csv'
    test_path = f'hdfs:/user/{netID}/{data_size}/test_df.csv'

    # schema (userId=148, movieId=44191, rating=4.0, timestamp=1482550089, row_number=1)
    train_df = spark.read.csv(train_path, header=True, quote='"', sep=",", inferSchema=True)
    val_df = spark.read.csv(val_path, header=True, quote='"', sep=",", inferSchema=True)
    test_df = spark.read.csv(test_path, header=True, quote='"', sep=",", inferSchema=True)
    # tune the hyper parameters
    rank_, reg_ = hyper_gridsearch(train_df, val_df)
    als = ALS(rank=rank_, maxIter=15, regParam=reg_, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    final_model = als.fit(train_df)
    final_model.save('als_model')

    # test model's performence on the testing set
    test_users = test_df.select('userId').distinct()
    userSubsetRecs = final_model.recommendForUserSubset(test_users, 100).select('userId', 'recommendations.movieId')
    true_recom_combined = test_df.groupby('userId').agg(collect_list('movieId').alias('true_item')).join(userSubsetRecs,
                                                                                                         on='userId',
                                                                                                         how='inner')
    predictionAndLabels = true_recom_combined.rdd.map(lambda row: (row[2], row[1]))
    metrics = RankingMetrics(predictionAndLabels)
    map_ = metrics.meanAveragePrecision
    precision_ = metrics.precisionAt(k)
    ndcg_ = metrics.ndcgAt(k)

    print(f"Test Set Final Performance: Map={map_}, Precision={precision_}, NDCG={ndcg_}")


def hyper_gridsearch(train_df, val_df):
    map_list, rank_list, reg_list = [], [], []
    if data_size == 'small':
        rank = [50, 100, 200, 300]
        regParam = [0.00001, 0.0001, 0.001,  0.01]
    if data_size == 'full':
        rank = [400, 500, 600]
        regParam = [0.00001, 0.0001, 0.001]
    grid = itertools.product(rank, regParam)
    map_lst = []
    for i in grid:
        als = ALS(rank=i[0], maxIter=30, regParam=i[1], userCol="userId", itemCol="movieId", ratingCol="rating",
                  coldStartStrategy="drop")
        model = als.fit(train_df)
        users = val_df.select('userId').distinct()
        userSubsetRecs = model.recommendForUserSubset(users, 100).select('userId', 'recommendations.movieId')
        true_recom_combined = val_df.groupby('userId').agg(collect_list('movieId').alias('true_item')).join(
            userSubsetRecs, on='userId', how='inner')
        predictionAndLabels = true_recom_combined.rdd.map(lambda row: (row[2], row[1]))
        metrics = RankingMetrics(predictionAndLabels)
        map_ = metrics.meanAveragePrecision
        print(f"Validation Set: Rank={i[0]},RegParam={i[1]},MAP={map_}")
        map_lst.append((map_, i[0], i[1]))
        rank_list.append(i[0])
        reg_list.append(i[1])
        map_list.append(map_)
    
    optimizer = py_builtin.max(map_lst)
    best_rank = optimizer[1]
    best_reg = optimizer[2]
    best_map = optimizer[0]
    print(f"Validation Set Best Performance: At rank={best_rank} and reg={best_reg}, max map={best_map}")
   
    df = pd.DataFrame(data={'rank':rank_list, 'reg':reg_list, 'val MAP':map_list})
    df.to_csv(f'als_{data_size}.csv')

    return best_rank, best_reg


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('Recomm_Sys_Data').getOrCreate()
    netID = getpass.getuser()
    data_size = sys.argv[1]
    # Call our main routine
    spark.sparkContext.setCheckpointDir(f'hdfs:/user/{netID}/checkpoints')
    main(spark, netID, data_size)
