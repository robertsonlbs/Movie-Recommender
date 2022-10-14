#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true evaluate.py

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.mllib.evaluation import RankingMetrics
import getpass
import sys
import itertools
import builtins as py_builtin
import pandas as pd

def main(spark, netID, k=100, rank_ = 100, reg_ = 0.001, prior = 100):
    
    train_path = f'hdfs:/user/{netID}/full/train_df.csv'
    train_df = spark.read.csv(train_path, header=True, quote='"', sep=",", inferSchema=True)
    test_path = f'hdfs:/user/{netID}/full/test_df.csv'
    test_df = spark.read.csv(test_path, header=True, quote='"', sep=",", inferSchema=True)

    # popularity
    items = train_df.groupby('movieId').agg(count('rating').alias('count'), sum('rating').alias('total_score'))
    items = items.select(col('movieId'), (col('total_score') / (col('count') + prior)).alias('popularity'))
    items.write.csv(f'hdfs:/user/{netID}/items_popularity',header=True)

    # ALS
    model = ALSModel.load(f"hdfs:/user/{netID}/als_full_model")
    items_matrix = model.itemFactors.rdd.sample(False, 0.1)
    items_matrix.saveAsTextFile(f'hdfs:/user/{netID}/items_factor')
    users_matrix = model.userFactors.rdd.sample(False, 0.01)
    users_matrix.saveAsTextFile(f'hdfs:/user/{netID}/users_factor')
    users = test_df.select('userId').distinct()
    userSubsetRecs = model.recommendForUserSubset(users, 500).select('userId', 'recommendations.movieId')
    true_recom_combined = test_df.groupby('userId').agg(collect_list('movieId').alias('true_item')).join(
            userSubsetRecs, on='userId', how='inner')
    true_recom_combined.write.csv(f'hdfs:/user/{netID}/predictions_500',header=True)

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('Recomm_Sys_Data').getOrCreate()
    netID = getpass.getuser()
    main(spark, netID)
