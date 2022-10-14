#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true model_out.py <small/full>

"""
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import getpass
import sys


def main(spark, netID, data_size):
    """
    Main routine for Data preprocessing and partitions
    Parameters
    ----------
    spark : SparkSession object
    """
    k = 100
    if data_size == 'small':
        train_path = f'hdfs:/user/{netID}/small/train_df.csv'
        model_path = f'hdfs:/user/{netID}/als_small_model'
    if data_size == 'full':
        train_path = f'hdfs:/user/{netID}/full/train_df.csv'
        model_path = f'hdfs:/user/{netID}/als_full_model'

    # schema (userId=148, movieId=44191, rating=4.0, timestamp=1482550089, row_number=1)
    train_df = spark.read.csv(train_path, header=True, quote='"', sep=",", inferSchema=True)
    rank_ = 200
    reg_ = 0.001
    max_iter = 20
    als = ALS(rank=rank_, maxIter=max_iter, regParam=reg_, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    final_model = als.fit(train_df)
    final_model.save(model_path)




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('Recomm_Sys_Data').getOrCreate()
    netID = getpass.getuser()
    data_size = sys.argv[1]
    # Call our main routine
    main(spark, netID, data_size)
