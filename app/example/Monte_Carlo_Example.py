# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         Monte_Carlo_Example
# Description:  蒙特卡罗方法模拟股票价格
# Author:       orange
# Date:         2021/5/5
# -------------------------------------------------------------------------------

import sys
import datetime
import os
from imp import reload

import matplotlib
import math
import numpy as np
from pyspark import SparkConf

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, DoubleType

reload(sys)


def createSpark(appName):
    conf = SparkConf().setAppName(appName)
    conf.set("spark.yarn.queue", "root.bpa.ad_push.kdd")
    #conf.set("spark.rdd.compress", "true")
    #conf.set("spark.broadcast.compress", "true")
    conf.set("spark.executor.instances", '200')
    conf.set("spark.executor.cores", '2') #spark.executor.cores：顾名思义这个参数是用来指定executor的分配更多的内核意味着executor并发能力越强，能够同时执行更多的task
    conf.set('spark.executor.memory', '32g')  #executor memory是每个节点上占用的内存。每一个节点可使用内存
    #conf.set('spark.default.parallelism','4000')
    #conf.set('spark.executor.memoryOverhead','10g')
    #conf.set('spark.dynamicAllocation.initialExecutors','200')
    #conf.set('spark.dynamicAllocation.minExecutors','100')
    #conf.set('spark.dynamicAllocation.maxExecutors','500')
    #conf.set("spark.sql.shuffle.partitions", "500") # 设置shuffle分区数
    #conf.set("spark.driver.maxResultSize", "5g")
    conf.set("spark.sql.hive.mergeFiles", "true")
    conf.set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "/usr/local/bin/python3.6")
    conf.set("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", "/usr/local/bin/python3.6")
    conf.set("spark.executorEnv.PYTHONPATH",
             "/usr/local/anaconda3/bin/python3.6/site-packages:/opt/spark-2.4.3-bin-hadoop2.6/python:/opt/spark-2.4.3-bin-hadoop2.6/python/lib/py4j-0.10.7-src.zip")
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    return spark


def multiVar(data):
    """ 方便利用已有函数进行cov,以及矩阵运算
    :param data:
    :return:
    """
    covarianceMatrix = np.mat(np.cov(data))
    means = data.mean(axis=1)
    dim = len(means)
    covMatEigenvalues, covMatEigenvectors = np.linalg.eig(covarianceMatrix)
    transposeMatrix = covMatEigenvectors.T
    tmpMatrix = np.copy(transposeMatrix)
    for row in range(0, dim):
        factor = np.sqrt(covMatEigenvalues[row])
        for col in range(0, dim):
            tmpMatrix[row, col] = tmpMatrix[row, col] * factor
    samplingMatrix = np.multiply(tmpMatrix, covMatEigenvectors)
    return means, covarianceMatrix, samplingMatrix


def trialReturns(seed, param, value, means, cov):
    pass


if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = "/usr/local/bin/python3.7"  # 该设置仅在当前程序有效 ,python3.7方式启动时才需
    pt_d=(datetime.datetime.now()+datetime.timedelta(days=-1)).strftime("%Y%m%d")
    pt_drop=(datetime.datetime.now()+datetime.timedelta(days=-5)).strftime("%Y%m%d")
    pt_h=None
    l=len(sys.argv)
    if l>=2:
        pt_d = sys.argv[1]
        if l>=3:
            pt_h = sys.argv[2]
    appName = "dwd_ad_push_device_info_df"
    sc = SparkConf
    spark = createSpark(appName=appName)

    # 读取全部因子数据文件
    data = spark.read.csv("./data/data.csv", header=True)
    data.registerTempTable("data")

    df = spark.sql('select '
                       'cast(label as double) label, '
                       'cast(WTI as double) WTI, '
                       'cast(WTI1 as double) WTI1, '
                       'cast(WTI2 as double) WTI2, '
                       'cast(Bond as double) Bond, '
                       'cast(Bond1 as double) Bond1,'
                       'cast(Bond2 as double) Bond2, '
                       'cast(GSPC as double) GSPC, '
                       'cast(GSPC1 as double) GSPC1, '
                       'cast(GSPC2 as double) GSPC2, '
                       'cast(NDAQ as double) NDAQ,'
                       'cast(NDAQ1 as double) NDAQ1, '
                       'cast(NDAQ2 as double) NDAQ2 '
                   'from data')
    # 由于计算需要使用WTI、Bond、GSPC、NDAQ这几个因子，因此先行处理
    factor = spark.sql('select cast(WTI as double) WTI, cast(Bond as double) Bond, cast(GSPC as double) GSPC, cast(NDAQ as double) NDAQ from data')
    list_factor = []
    list_factor.append(factor.select('WTI').rdd.max(lambda x: x[0]).collect())
    list_factor.append(factor.select('Bond').rdd.max(lambda x: x[0]).collect())
    list_factor.append(factor.select('GSPC').rdd.max(lambda x: x[0]).collect())
    list_factor.append(factor.select('NDAQ').rdd.max(lambda x: x[0]).collect())

    print("全部因子预览")
    df.show(5)
    print(list_factor)

    # 数据准备好之后开始使用OLS最小二乘法进行多元线性回归
    vecAssebler = VectorAssembler(inputCols=["WTI", "WTI1", "WTI2", "Bond", "Bond1", "Bond2", "GSPC", "GSPC1", "GSPC2", "NDAQ", "NDAQ1", "NDAQ2"])
    vecDF = vecAssebler.transform(df)
    lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal")
    model = lr.fit(vecDF)
    factorWeights = []

    # 将模型系数保存到factorWeights列表中
    factorWeights.append(model.coefficients)

    # 进行蒙特卡罗模拟
    data = np.array(list_factor)
    # 计算均值和cov
    means, cov, samplingMatrix = multiVar(data)
    numTrials = 100
    bFactorWeights = sc.broadcast(factorWeights)
    parallelism = 100
    baseSeed = 1496
    seeds = []
    for i in range(baseSeed, baseSeed + parallelism):
        seeds.append(i)
    seedRdd = sc.parallelize(seeds, parallelism)
    trials = seedRdd.flatMap(lambda seed: trialReturns(seed, numTrials/parallelism, bFactorWeights.value, means, cov))
    size = trials.count()

    # VaR And CVaR
    test = trials.takeOrdered(max(size/20), 1)
    VaR = test[-1]
    CVaR = sum(test)/len(test)
    print(VaR)
    print(CVaR)


