# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         TumblingWindow_Example001
# Description:
# Author:       orange
# Date:         2021/6/11
# -------------------------------------------------------------------------------


from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import from_json, col, to_timestamp, window, expr, sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from common.log.logger import Log4j


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("File Streaming Demo") \
        .master("local[3]") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.streaming.schemaInference", "true") \
        .getOrCreate()

    logger = Log4j(spark)

    stock_schema = StructType([
        StructField("CreatedTime", StringType()),
        StructField("Type", StringType()),
        StructField("Amount", IntegerType()),
        StructField("BrokerCode", StringType())
    ])

    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "test") \
        .option("startingOffsets", "earliest") \
        .load()

    value_df = kafka_df.select(from_json(col("value").cast("string"), stock_schema).alias("value"))

    trade_df = value_df.select("value.*") \
        .withColumn("CreatedTime", to_timestamp(col("CreatedTime"), "yyyy-MM-dd HH:mm:ss")) \
        .withColumn("Buy", expr("case when Type == 'BUY' then Amount else 0 end")) \
        .withColumn("Sell", expr("case when Type == 'SELL' then Amount else 0 end"))

    window_agg_df = trade_df \
        .groupBy(  # col("BrokerCode"),
        window(col("CreatedTime"), "15 minute")) \
        .agg(sum("Buy").alias("TotalBuy"),
             sum("Sell").alias("TotalSell"))

    output_df = window_agg_df.select("window.start", "window.end", "TotalBuy", "TotalSell")

    '''
    running_total_window = Window.orderBy("end") \
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    final_output_df = output_df \
        .withColumn("RTotalBuy", sum("TotalBuy").over(running_total_window)) \
        .withColumn("RTotalSell", sum("TotalSell").over(running_total_window)) \
        .withColumn("NetValue", expr("RTotalBuy - RTotalSell"))
    final_output_df.show(truncate=False)
    '''
    window_query = output_df.writeStream \
        .format("console") \
        .outputMode("update") \
        .option("checkpointLocation", "chk-point-dir") \
        .trigger(processingTime="1 minute") \
        .start()

    logger.info("Waiting for Query")
    window_query.awaitTermination()