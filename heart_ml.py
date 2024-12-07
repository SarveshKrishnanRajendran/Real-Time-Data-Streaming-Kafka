#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col


# In[2]:


from pyspark.sql.types import StructType, StructField, DoubleType,IntegerType
# Step 1: Create a Spark session
spark = SparkSession.builder.appName("Heart-Disease-Classification").getOrCreate()
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
    StructField("ST slope", DoubleType(), True),
    StructField("target", DoubleType(), True)
])
# Step 2: Load the Iris dataset
# Assuming iris.csv is in the current directory
heart_df = spark.read.csv("heart_statlog_cleveland_hungary_final.csv", header=True, schema=schema)


# In[3]:


heart_df.printSchema()


# In[4]:


train_df, test_df = heart_df.randomSplit([0.8, 0.2], seed=42)


# In[ ]:





# In[5]:


heart_df.count()


# In[6]:


heart_df.printSchema()


# In[7]:


heart_df.select("chest pain type").distinct().show()
heart_df.select("resting ecg").distinct().show()
heart_df.select("ST slope").distinct().show()


# In[8]:


from pyspark.sql.functions import col, sum
null_check = heart_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in heart_df.columns])
null_check.show()


# In[9]:


# train_df, test_df = heart_df.randomSplit([0.8, 0.2], seed=42)


# In[10]:


categorical_columns = ["chest pain type","resting ecg","ST slope"]


# In[11]:


other_columns = ["age","sex","resting bp s","cholesterol","fasting blood sugar","max heart rate","exercise angina","oldpeak"]


# In[12]:


string_indexer_col = [i+"index" for i in categorical_columns]


# In[13]:


string_indexer_col


# In[14]:


ohe_col = [j+"ohe" for j in categorical_columns]


# In[15]:


ohe_col


# In[16]:


from pyspark.ml.feature import OneHotEncoder,StringIndexer
# Correct way to define StringIndexer and OneHotEncoder
string_indexer = [StringIndexer(inputCol=col, outputCol=col + "index", handleInvalid="skip") for col in categorical_columns]
ohe = [OneHotEncoder(inputCol=col + "index", outputCol=col + "ohe", dropLast=True, handleInvalid="keep") for col in categorical_columns]


# In[17]:


assembler = VectorAssembler(handleInvalid="skip",inputCols=ohe_col + other_columns, outputCol="features")


# In[18]:


from pyspark.ml.classification import LogisticRegression

# Define the model
lr = LogisticRegression(featuresCol="features", labelCol="target", maxIter=10)


# In[19]:


from pyspark.ml import Pipeline
# Step 2: Create a pipeline to apply all transformations sequentially
pipline_model = Pipeline(stages=string_indexer + ohe + [assembler,lr])

pipline = pipline_model.fit(train_df)


# In[20]:


data = pipline.transform(train_df).select("features", col("target"))


# In[21]:


data.show()


# In[22]:


train_df.count()


# In[23]:


test_df.count()


# In[24]:


data.count()


# In[25]:


982+208


# In[26]:


predictions = pipline.transform(test_df)

# Show some predictions (you can specify more columns if you want)
predictions.select("features", "target", "prediction", "probability").show(5)


# In[27]:


predictions.printSchema()


# In[28]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize evaluators for Precision, Recall, and F1-Score
precision_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="precisionByLabel")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="recallByLabel")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")


# In[29]:


# Set the positive class label (1 or 0 depending on your dataset's labeling)
positive_class = 1

# Calculate precision, recall, and F1-score
precision = precision_evaluator.evaluate(predictions, {precision_evaluator.metricLabel: positive_class})
recall = recall_evaluator.evaluate(predictions, {recall_evaluator.metricLabel: positive_class})
f1_score = f1_evaluator.evaluate(predictions)

# Print the results
print(f"Precision for positive class (label = {positive_class}): {precision}")
print(f"Recall for positive class (label = {positive_class}): {recall}")
print(f"F1-Score: {f1_score}")


# In[30]:


pipline.write().overwrite().save("pipeline")


# In[31]:


# Stop the Spark session
spark.stop()


# In[ ]:




