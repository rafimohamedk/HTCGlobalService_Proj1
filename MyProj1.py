# Databricks notebook source
# MAGIC %md
# MAGIC ### Project Name: 

# COMMAND ----------

# MAGIC %md
# MAGIC ###	Problem Statement:
# MAGIC ###### Given 3 datasets related to online products sales in the year 2019 for months Jan, Feb and March for a given store.
# MAGIC ###### Need to do data engineering using PySpark or Spark(Scala/Java) on these datasets to obtain the following objectives **
# MAGIC ###### 1.	Cleanse the data removing blank rows
# MAGIC ###### 2.	Get the date on which max sales was done by product in these 3 months
# MAGIC ###### 3.	Get the date on which max sales was done for all products in these 3 months
# MAGIC ###### 4.	Get the average sales value for each product in these 3 months
# MAGIC ###### 5.	Create a combined dataset merging all these 3 datasets with order by date in desc order and add a new column which is “salesdiff” where this column will contain the difference of the sales in the current row (current date of that row) and the next row (previous date of that row, as the date columns are sorted by desc) grouped on the product
# MAGIC ###### For the last row, next row will be blank so consider the sales as 0
# MAGIC ###### 6.	Get the orderId and purchase address details who made max sales in all the 3 months
# MAGIC ###### 7.	Extract city from the purchase address column which is 2nd element in , delimited separated string and determine the city from where more orders came in all these 3 months
# MAGIC ###### 8.	Get the total order count details for each city in all the 3 months
# MAGIC ###### 9.	Create partition by “bi-weekly” order dates and save the data in physical storage in DBFS or ADLS or local storage or delta table or S3 or HDFS.
# MAGIC
# MAGIC ###### Note: 
# MAGIC ###### •	Sales value calculated by qty * price
# MAGIC ###### •	orders count can be determined based on orderId(one orderId means 1 order)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Importing necessary Libraries

# COMMAND ----------

# Importing required libraries
import pyspark.pandas as ps
import datetime as dt
import pandas as pd
from datetime import timedelta
from pyspark.sql.types import *
from pyspark.sql import * 
from pyspark.sql.functions import *
from functools import *

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading the files

# COMMAND ----------

mnt = "dbfs:/FileStore"
dbutils.fs.ls(mnt)

# COMMAND ----------

Sales_Jan = spark.read.option("header", True).csv("dbfs:/FileStore/Sales_January_2019.csv")
Sales_Feb = spark.read.option("header", True).csv("dbfs:/FileStore/Sales_February_2019.csv")
Sales_Mar = spark.read.option("header", True).csv("dbfs:/FileStore/Sales_March_2019.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating New column for sales month as File_month and for sales year as File_Year

# COMMAND ----------

Sales_Jan1 = Sales_Jan.withColumn("File_month", lit("Jan")).withColumn("File_Year", lit("2019"))
Sales_Feb1 = Sales_Feb.withColumn("File_month", lit("Feb")).withColumn("File_Year", lit("2019"))
Sales_Mar1 = Sales_Mar.withColumn("File_month", lit("Mar")).withColumn("File_Year", lit("2019"))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Combining all 3 months df into a single df. 

# COMMAND ----------

# Combining all the 3 dataset for further analysis

Combined_Sales = Sales_Jan1.union(Sales_Feb1).union(Sales_Mar1)

# COMMAND ----------

# Generate the data quality report before analysis for the final comparison
# Calculate total count and null count per FileMonth and FileYear

# Calculate the null counts for each column
null_counts = Combined_Sales.select([sum(col(c).isNull().cast("int")).alias(c) for c in Combined_Sales.columns])
null_counts = null_counts.withColumn("metric", lit("null_count"))

# Calculate the total counts for each column
total_counts = Combined_Sales.select([count(col(c)).alias(c) for c in Combined_Sales.columns])
total_counts = total_counts.withColumn("metric", lit("total_count"))

# Combine the null counts and total counts into a single DataFrame
Combined_Sales_dq_report = null_counts.union(total_counts).select("metric", *Combined_Sales.columns)



# COMMAND ----------

Combined_Sales_dq_report.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.	Cleanse the data removing blank rows 

# COMMAND ----------

# MAGIC %md
# MAGIC #### As Order ID and Product are the key columns for analysis dropping the rows with no value

# COMMAND ----------

Combined_Sales.filter(~col("Order ID").rlike('^[0-9]+$')).display()
Combined_Sales_incorr_ordid_dropped = Combined_Sales.filter(col("Order ID").rlike('^[0-9]+$')).filter(col("Order ID").isNotNull())

# COMMAND ----------

# Validating if new df contains the incorrect OrderID values and Null values
Combined_Sales_incorr_ordid_dropped.filter(~col("Order ID").rlike('^[0-9]+$')).display()
Combined_Sales_incorr_ordid_dropped.filter(col("Order ID").isNull()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adding Total bill column based on quantity * Price Each

# COMMAND ----------

Combined_Sales_Total_bill_added = Combined_Sales_incorr_ordid_dropped \
                                    .withColumn("Price Each", col("Price Each").cast("float")) \
                                    .withColumn("Total_Bill", col("Quantity Ordered") * col("Price Each"))

# COMMAND ----------

Combined_Sales_Total_bill_added.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Checking unique counts of Product ID across each month

# COMMAND ----------

Combined_Sales_Total_bill_added.groupBy("Product", "File_month").count().groupBy("File_month").count().display()

# COMMAND ----------

# Converting date in string format to date format

Combined_Sales_date_format = Combined_Sales_Total_bill_added.withColumn("Order Date1", to_date(to_timestamp(col("Order Date"), "MM/dd/yy HH:mm"), "yyyy-MM-dd"))

# COMMAND ----------

Combined_Sales_date_format.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Get the date on which max sales was done by product in these 3 months

# COMMAND ----------

# Fetching  the date on which the max sales happened in 3 months

window_spec = Window.orderBy(col("Total_Sales").desc())

Combined_Sales_max_sales_by_product = Combined_Sales_date_format.groupBy("Product","Order Date1") \
                            .agg(sum("Total_Bill").alias("Total_Sales")) \
                            .orderBy("Total_Sales", ascending=False) \
                            .withColumn("Dense_Rank", dense_rank().over(window_spec)) \
                            .filter(col("Dense_Rank") == 1)
Combined_Sales_max_sales_by_product.drop("Dense_Rank").display()

# COMMAND ----------

# Fetching the date in which top sales happened in each month

Combined_Sales_max_sales = Combined_Sales_date_format \
                        .groupBy("Product", "File_month","Order Date1").agg(sum("Total_Bill").alias("Total_Sales")).orderBy("Total_Sales", ascending=False)

avail_months = [row["File_month"] for row in Combined_Sales_max_sales.select("File_month").distinct().collect()]
for month in avail_months:
    max_val = [row["Max_sales"] for row in Combined_Sales_max_sales.filter(col("File_month") == month).agg(max("Total_Sales").alias("Max_sales")).collect()][0]
    print("Max_sales_in", month, "is", max_val)

# Create a window specification
window_spec = Window.partitionBy("File_month").orderBy(col("Total_Sales").desc())
Combined_Sales_max_sales_with_dense_rank = Combined_Sales_max_sales.withColumn("dense_rank", dense_rank().over(window_spec))

# Filter to keep only the row with the max sales per product
max_sales_date_per_product = Combined_Sales_max_sales_with_dense_rank.filter(col("dense_rank") == 1).select("File_month","Product", "Order Date1", "Total_Sales").orderBy("Total_Sales")

max_sales_date_per_product.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.	Get the date on which max sales was done for all products in these 3 months

# COMMAND ----------

# Fetching the date on which max sales done for all products in 3 months

window_spec = Window.orderBy(col("Total_Sales").desc())

Combined_Sales_max_sales_all_product = Combined_Sales_date_format.groupBy("Order Date1") \
                            .agg(sum("Total_Bill").alias("Total_Sales")) \
                            .orderBy("Total_Sales", ascending=False) \
                            .withColumn("Dense_Rank", dense_rank().over(window_spec)) \
                            .filter(col("Dense_Rank") == 1)
Combined_Sales_max_sales_all_product.drop("Dense_Rank").display()

# COMMAND ----------

# Fetching the max sales done for all products in each month

Combined_Sales_max_sales_all_prod = Combined_Sales_date_format.groupBy("File_month","Order Date1").agg(sum("Total_Bill").alias("Total_Sales")).orderBy("Total_Sales", ascending=False)

window_spec = Window.partitionBy("File_month").orderBy(col("Total_Sales").desc())
Combined_Sales_max_sales_all_prod_with_dense_rank = Combined_Sales_max_sales_all_prod.withColumn("dense_rank", dense_rank().over(window_spec))

# Filter to keep only the row with the max sales per product
Combined_Sales_max_sales_all_prod_all_date = Combined_Sales_max_sales_all_prod_with_dense_rank.filter(col("dense_rank") == 1).select("File_month","Order Date1", "Total_Sales")

# Show the results
Combined_Sales_max_sales_all_prod_all_date.orderBy(col("Total_Sales").desc()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.	Get the average sales value for each product in these 3 months

# COMMAND ----------

# Finding Average sales for each product

Combined_Sales_max_sales_all_prod_avg_sales = Combined_Sales_date_format.groupBy("Product") \
                                                .agg((sum(col("Total_Bill")) / sum(col("Quantity Ordered"))).alias("Average_Sales")) \
                                                .orderBy(col("Average_Sales").desc())

Combined_Sales_max_sales_all_prod_avg_sales.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.	Create a combined dataset merging all these 3 datasets with order by date in desc order and add a new column which is “salesdiff” where this column will contain the difference of the sales in the current row (current date of that row) and the next row (previous date of that row, as the date columns are sorted by desc) grouped on the product
# MAGIC #### For the last row, next row will be blank so consider the sales as 0
# MAGIC

# COMMAND ----------

window_spec = Window.partitionBy("Product").orderBy(col("Order Date1").desc())

Combined_Sales_date_ordered_desc = Combined_Sales_date_format.withColumn("Row_Number", row_number().over(window_spec))
Combined_Sales_date_ordered_desc_with_sales_diff = Combined_Sales_date_ordered_desc \
                                                .withColumn("salesdiff", col("Total_Bill") - lag(col("Total_Bill"), 1).over(window_spec)) \
                                                .withColumn("salesdiff", when(col("salesdiff").isNull(), lit(0)).otherwise("salesdiff"))

Combined_Sales_date_ordered_desc_with_sales_diff.display()


# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.	Get the orderId and purchase address details who made max sales in all the 3 months

# COMMAND ----------

window_spec = Window.orderBy(col("Total_Sales").desc())

Combined_Sales_max_sales_ordid_add = Combined_Sales_date_format.groupBy("Order ID").agg(sum("Total_Bill").alias("Total_Sales")).orderBy("Total_Sales", ascending=False).withColumn("Dense_Rank", dense_rank().over(window_spec))

maximum_sales_ord_id = [row["Order ID"] for row in Combined_Sales_max_sales_ordid_add.filter(col("Dense_Rank") == 1).select("Order ID").collect()][0]

Combined_Sales_date_format.filter(col("Order ID") == maximum_sales_ord_id).select("Order ID", "Purchase Address").display()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC #### 7.	Extract city from the purchase address column which is 2nd element in , delimited separated string and determine the city from where more orders came in all these 3 months

# COMMAND ----------

window_spec = Window.orderBy(col("count").desc())

Combined_Sales_date_format_with_city = Combined_Sales_date_format.withColumn("City", split(col("Purchase Address"), ",")[1]) \
                                                                 .groupBy("City").count() \
                                                                 .withColumn("Dense_Rank", dense_rank().over(window_spec)) \
                                                                 .filter(col("Dense_Rank") == 1)

Combined_Sales_date_format_with_city.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8.	Get the total order count details for each city in all the 3 month

# COMMAND ----------

Combined_Sales_date_format_with_city_sales_per_month = Combined_Sales_date_format \
                                                        .withColumn("City", split(col("Purchase Address"), ",")[1]) \
                                                        .groupBy("City").count()

Combined_Sales_date_format_with_city_sales_per_month.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 9.	Create partition by “bi-weekly” order dates and save the data in physical storage in DBFS or ADLS or local storage or delta table or S3 or HDFS.

# COMMAND ----------

# from pyspark.sql import functions as F

Combined_Sales_with_biweekly_partitioned = Combined_Sales_date_ordered_desc_with_sales_diff \
                                            .withColumn("bi_weekly_partition", concat(year("Order Date1"), lit("_"), floor((dayofyear("Order Date1") - 1) / 14)))

# COMMAND ----------

write_path = "dbfs:/FileStore/Sales/"
Combined_Sales_with_biweekly_partitioned.write.partitionBy("bi_weekly_partition").mode("overwrite").parquet(write_path)

# COMMAND ----------

dbutils.fs.ls(write_path)

# COMMAND ----------

df_2019_1 = spark.read.format("parquet").load("dbfs:/FileStore/Sales/bi_weekly_partition=2019_1/")
df_2019_1.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### ------------------------------------------------------End of Project------------------------------------------------------
