#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true data_pre.py <small/full>

'''
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import random
import getpass
import sys


def main(spark, netID, data_size):
    '''
    Main routine for Data preprocessing and partitions
    Parameters
    ----------
    spark : SparkSession object
    '''
    # Note: 
    # In small-dataset, every user rates at least 20 movies.
    # In full-dataset, every user rates at least 1 movie.
    if data_size == 'small':
        ratings_path = f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv'
        ratings_df = spark.read.csv(ratings_path, header=True, quote='"', sep=",", inferSchema=True)
        # Take distinc userId from the ratings
        userId_lst = list(ratings_df.select('userId').distinct().toPandas()['userId'])
        # Partition userId list into train, validation, test
        random.shuffle(userId_lst)
        train_user, val_user, test_user = train_val_test_small(userId_lst, 0.6, 0.2, 0.2)
    if data_size == 'full':
        ratings_path = f'hdfs:/user/{netID}/movielens/ml-latest/ratings.csv'
        ratings_df = spark.read.csv(ratings_path, header=True, quote='"', sep=",", inferSchema=True)
        train_user, val_user, test_user = train_val_test_full(ratings_df, 0.6, 0.2, 0.2)

    held_out_num = 3
    train_df, val_df, test_df = partitions(ratings_df, train_user, val_user, test_user, held_out_num)

    if data_size == 'small':
        train_df.write.csv(f'hdfs:/user/{netID}/small/train_df.csv', header=True)
        val_df.write.csv(f'hdfs:/user/{netID}/small/val_df.csv', header=True)
        test_df.write.csv(f'hdfs:/user/{netID}/small/test_df.csv', header=True)
    if data_size == 'full':
        train_df.write.csv(f'hdfs:/user/{netID}/full/train_df.csv', header=True)
        val_df.write.csv(f'hdfs:/user/{netID}/full/val_df.csv', header=True)
        test_df.write.csv(f'hdfs:/user/{netID}/full/test_df.csv', header=True)


def partitions(ratings_df, train_user, val_user, test_user, hold_out_num):
    '''
    Parameters
    ----------  
    rating_df : Spark dataframe
    hold_out_num : num to hold out for validation and testing
    ** : User identifiers(list of user id)

    Return
    ----------
    Train/Validation/Test Spark dataframes  
    '''
    hold_out = list(range(1, hold_out_num + 1))
    window = Window.partitionBy('userId').orderBy(col('timestamp').desc())
    ratings_df_rowNum = ratings_df.withColumn("row_number", row_number().over(window))
    train_df = ratings_df_rowNum.filter((col('userId').isin(train_user)) | (~col('row_number').isin(hold_out)))
    val_df = ratings_df_rowNum.filter((col('userId').isin(val_user)) & (col('row_number').isin(hold_out)))
    test_df = ratings_df_rowNum.filter((col('userId').isin(test_user)) & (col('row_number').isin(hold_out)))
    return train_df, val_df, test_df


def train_val_test_small(userId_lst, train_size, val_size, test_size):
    '''
    Output userId list corresponding to train, val, and test
    '''
    try:
        train_size + val_size + test_size == 1.0
    except:
        print('size does not sum to 1')

    n = len(userId_lst)
    train_num, val_num = int(n * train_size), int(n * val_size)
    test_num = n - train_num - val_num
    return userId_lst[:train_num], userId_lst[train_num:train_num + val_num], userId_lst[train_num + val_num:]


def train_val_test_full(ratings_df, train_size, val_size, test_size):
    '''
    Output userId list corresponding to train, val, and test
    '''
    try:
        train_size + val_size + test_size == 1.0
    except:
        print('size does not sum to 1')

    # Partition userId list into train, validation, test
    user_count = ratings_df.groupby('userId').count()
    infrequent_user_lst = list(user_count.filter(col('count') <= 4).select('userId').toPandas()['userId'])
    frequent_user_lst = list(user_count.filter(col('count') > 4).select('userId').toPandas()['userId'])

    n = len(infrequent_user_lst) + len(frequent_user_lst)
    val_num, test_num = int(n * val_size), int(n * test_size)
    val_lst = frequent_user_lst[:val_num]
    test_lst = frequent_user_lst[val_num:val_num + test_num]
    train_lst = frequent_user_lst[val_num + test_num:]
    train_lst.extend(infrequent_user_lst)
    return train_lst, val_lst, test_lst


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('Recomm_Sys_Data').getOrCreate()
    netID = getpass.getuser()
    data_size = sys.argv[1]
    # Call our main routine
    main(spark, netID, data_size)
