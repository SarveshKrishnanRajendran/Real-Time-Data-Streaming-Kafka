#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, DoubleType,IntegerType

#Creating a Spark session with Kafka configuration
spark = SparkSession.builder \
    .appName("Kafka-Heart-Test") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
    .getOrCreate()

#Defining schema for incoming data
schema = StructType([
    StructField("age", DoubleType(), True),
    StructField("sex", DoubleType(), True),
    StructField("chest pain type", DoubleType(), True),
    StructField("resting bp s", DoubleType(), True),
    StructField("cholesterol", DoubleType(), True),
    StructField("fasting blood sugar", DoubleType(), True),
    StructField("resting ecg", DoubleType(), True),
    StructField("max heart rate", DoubleType(), True),
    StructField("exercise angina", DoubleType(), True),
    StructField("oldpeak", DoubleType(), True),
    StructField("ST slope", DoubleType(), True)
])
#Reading data from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "heartinput") \
    .option("failOnDataLoss", "false") \
    .load()

#Printing the schema of kafka_df
kafka_df.printSchema()


# In[2]:

#Parsing the JSON data from Kafka
from pyspark.sql.functions import col, from_json
value_df = kafka_df.selectExpr("CAST(value AS STRING)as json_value")
parsed_df = value_df.select(from_json(col("json_value"), schema).alias("data")).select("data.*")
parsed_df.printSchema()


# In[3]:


# value_df.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start() \
#     .awaitTermination()


# In[4]:


# parsed_df.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()


# In[5]:

#Apply pre-trained ML model for predictions
from pyspark.ml import PipelineModel
pipeline_model = PipelineModel.load("/Users/sarveshkrishnan/Desktop/termpaper/heartdisease/pipeline")

predicted = pipeline_model.transform(parsed_df)


# In[6]:


selected_predictions = predicted.select("age", "sex", "prediction")


# In[8]:


# selected_predictions.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()


# In[9]:
from pyspark.sql.functions import when, col

# Creating a new column with a descriptive result based on the prediction
output = selected_predictions.withColumn(
    "result",
    when(col("prediction") == 1, "Risk of Heart Disease")
    .otherwise("No Risk of Heart Disease")
)

# Streaming the results to Kafka
query = output.selectExpr("CAST(result AS STRING) AS value") \
    .writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "heartoutput") \
    .option("checkpointLocation", "/Users/sarveshkrishnan/spark-checkpoints/sentimentoutput") \
    .start()

query.awaitTermination()




# In[ ]:




