#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

#Creating a Spark Session with Kafka configurations
spark = SparkSession.builder \
    .appName("KafkaSentimentAnalysis") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()


# In[2]:

#Reading data from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "sentimentinput") \
    .option("failOnDataLoss", "false") \
    .load()

# Convert the Kafka message key and value from bytes to strings
kafka_string_df = kafka_df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING) as review")


# In[3]:


# kafka_df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)").writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .start() \
#     .awaitTermination()


# In[4]:


from pyspark.sql.functions import col

#Converting the Kafka message from bytes to strings (assuming the message is plain text)
parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as review")

#Displaying the schema of the parsed_df
parsed_df.printSchema()

# Write the parsed DataFrame to the console
# parsed_df.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .start() \
#     .awaitTermination()


# In[5]:


#Loading the preprocessing pipeline model
pipeline_model = PipelineModel.load("/Users/sarveshkrishnan/Desktop/termpaper/sentiment/preprocess")


# In[6]:


df=pipeline_model.transform(parsed_df)


# In[7]:


# df.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .start()


# In[8]:

#loading the pretrained model
from pyspark.ml.classification import LogisticRegressionModel
loaded_model = LogisticRegressionModel.load("/Users/sarveshkrishnan/Desktop/termpaper/sentiment/saved_model")


# In[9]:
final_predictions = loaded_model.transform(df)
# In[10]:


# final_predictions \
#     .writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .start()


# In[11]:
from pyspark.sql.functions import when, col

# Creating a new column with a descriptive result based on the prediction
output = final_predictions.withColumn(
    "result",
    when(col("prediction") == 1, "Positive Sentiment")
    .otherwise("Negative Sentiment")
)

# Streaming the results to Kafka
query = output.selectExpr("CAST(result AS STRING) AS value") \
    .writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "sentimentoutput") \
    .option("checkpointLocation", "/Users/sarveshkrishnan/spark-checkpoints/sentimentoutput") \
    .start()

query.awaitTermination()

# In[ ]:




