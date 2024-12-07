#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[2]:


spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .getOrCreate()


# In[3]:


schema = StructType([
    StructField("review", StringType(), True),
    StructField("label", IntegerType(), True)
])


# In[4]:


# Load the dataset with the defined schema
data = spark.read.csv("reviews_sentiment.csv", schema=schema, header=True)
data.show()


# In[5]:


data.count()


# In[6]:


data=data.dropna()


# In[7]:


data.count()


# In[8]:


# Tokenize the text
tokenizer = Tokenizer(inputCol="review", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Convert words to feature vectors using CountVectorizer
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")

# Calculate TF-IDF
idf = IDF(inputCol="raw_features", outputCol="features")


# In[9]:


# Create a pipeline
pipelinemodel = Pipeline(stages=[tokenizer, remover, vectorizer, idf])


# In[10]:


pipeline= pipelinemodel.fit(data)


# In[11]:


df=pipeline.transform(data).select("features","label")


# In[12]:


df.show()


# In[13]:


# Split the data into training and testing datasets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)


# In[14]:


lr = LogisticRegression(featuresCol="features", labelCol="label")


# In[15]:


model=lr.fit(train_data)


# In[16]:


predictions=model.transform(test_data)


# In[17]:


predictions.show()


# In[18]:


predictions.select("features", "label", "prediction", "probability").show(5)


# In[19]:


predictions.printSchema()


# In[20]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize evaluators for Precision, Recall, and F1-Score
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")


# In[21]:


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


# In[23]:


model.save("saved_model")


# In[24]:


pipeline.write().overwrite().save("preprocess")


# In[25]:


spark.stop()


# In[ ]:




