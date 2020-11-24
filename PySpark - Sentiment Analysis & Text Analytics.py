# Databricks notebook source
# MAGIC %md
# MAGIC ###### Notebook by Jake Chen (Team Richmond)

# COMMAND ----------

#Import master data

df_eda = spark.sql("select * from mma2021w_richmond.df_eda")

display(df_eda)

# COMMAND ----------

#TF-IDF

final_df = df_eda
final_text = 'unigrams'

#CountVectorizer
#minDF specifies in how many rows does a word need to appear for it to be counted, used for removing terms that appear too infrequently
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol=final_text, outputCol="rawFeatures",minDF = 10)
cv_model = cv.fit(final_df)
final_df = cv_model.transform(final_df)

#Apply the IDF part of TF-IDF (term frequency–inverse document frequency)

from pyspark.ml.feature import  IDF

#IDF down-weighs features which appear frequently in a corpus. This generally improves performance when using text as features since most frequent, and hence less important words, get down-weighed.
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(final_df)
final_df = idfModel.transform(final_df)


# COMMAND ----------

#Remove all netural rating, that is, overall=3 

df_sentiment = final_df.filter("overall !=3")



#Group the overall score into sentiment labels

from pyspark.ml.feature import Bucketizer

splits = [-float("inf"), 4.0, float("inf")]
bucketizer = Bucketizer(splits=splits, inputCol="overall", outputCol="label_overall")

df_sentimentbucket = bucketizer.transform(df_sentiment)

#Show results and count of split
#df_sentimentbucket.groupBy("overall","label_overall").count().show()

# COMMAND ----------

#Check target imbalance

dataset_size=float(df_sentimentbucket.select("label_overall").count())
numPositives=df_sentimentbucket.select("label_overall").where('label_overall == 1').count()
per_ones=(float(numPositives)/float(dataset_size))*100
numNegatives=float(dataset_size-numPositives)
print('The number of 1s are {}'.format(numPositives))
print('Percentage of 1s are {}'.format(per_ones))


#Check existing imbalance

from pyspark.sql.functions import lower, col

major_df = df_sentimentbucket.filter(col("label_overall") == 1)
minor_df = df_sentimentbucket.filter(col("label_overall") == 0)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))

# COMMAND ----------

#Perform undersampling because of large data size

sampled_majority_df = major_df.sample(False, 1/ratio, seed=1234)
df_sentimentbucket_balance = sampled_majority_df.unionAll(minor_df)

#Check updated balance
new_major_df = df_sentimentbucket_balance.filter(col("label_overall") == 1)
new_minor_df = df_sentimentbucket_balance.filter(col("label_overall") == 0)
new_ratio = int(new_major_df.count()/new_minor_df.count())
print("ratio: {}".format(new_ratio))

#Check by overall score
df_sentimentbucket_balance.groupBy("overall","label_overall").count().show()
#Make sure ratio = 1

# COMMAND ----------

#split into training and test dataset
train_sentiment, test_sentiment = df_sentimentbucket_balance.randomSplit([0.8, 0.2], seed=12345)

#Check imbalance in train set
train_dataset_size=float(train_sentiment.select("label_overall").count())
train_numPositives=train_sentiment.select("label_overall").where('label_overall == 1').count()
train_per_ones=(float(train_numPositives)/float(train_dataset_size))*100
train_numNegatives=float(train_dataset_size-train_numPositives)
print('The number of 1s in train are {}'.format(train_numPositives))
print('Percentage of 1s in train are {}'.format(train_per_ones))

#Check imbalance in test set
test_dataset_size=float(test_sentiment.select("label_overall").count())
test_numPositives=test_sentiment.select("label_overall").where('label_overall == 1').count()
test_per_ones=(float(test_numPositives)/float(test_dataset_size))*100
test_numNegatives=float(test_dataset_size-test_numPositives)
print('The number of 1s in test are {}'.format(test_numPositives))
print('Percentage of 1s in test are {}'.format(test_per_ones))


# COMMAND ----------

#Train Model
from pyspark.ml.classification import LogisticRegression

lambda_par = 0.02
alpha_par = 0.1

lr = LogisticRegression(labelCol="label_overall", featuresCol="features",maxIter=20, regParam=lambda_par, elasticNetParam=alpha_par)
lr_model=lr.fit(train_sentiment)

predict_train=lr_model.transform(train_sentiment)

# COMMAND ----------

#Estimating accuracy

from pyspark.sql import functions as fn

lr_model.transform(test_sentiment).\
    select(fn.expr('float(prediction = label_overall)').alias('correct')).\
    select(fn.avg('correct')).show()

#Shows 0.776295668948643 from alpha = 0.2

# COMMAND ----------

#Create a table of words with their weights

import pandas as pd
vocabulary = cv_model.vocabulary
weights = lr_model.coefficients.toArray()
coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})

#Most negative words
df_negative_sentiment = coeffs_df.sort_values('weight', ascending=True).head(100)

display(df_negative_sentiment)


# COMMAND ----------

#Most positive words
df_positive_sentiment = coeffs_df.sort_values('weight', ascending=False).head(100)

display(df_positive_sentiment)

# COMMAND ----------

#Append two sentiment dataset
df_sentiment = df_positive_sentiment.append(df_negative_sentiment, ignore_index=True)

display(df_sentiment)

# COMMAND ----------



# COMMAND ----------

#Parameter tunning to find optimal lambda_par and alpha_par

from pyspark.ml.tuning import ParamGridBuilder

grid = ParamGridBuilder().\
    addGrid(lr_model.regParam, [0., 0.01, 0.02]).\
    addGrid(lr_model.elasticNetParam, [0., 0.2, 0.4]).\
    build()

all_models = []
for j in range(len(grid)):
    print("Fitting model {}".format(j+1))
    reg_model = lr.fit(train_sentiment, grid[j])
    all_models.append(reg_model)


# COMMAND ----------

accuracies = [m.\
    transform(test_sentiment).\
    select(fn.avg(fn.expr('float(label_overall = prediction)')).alias('accuracy')).\
    first().\
    accuracy for m in all_models]

accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling Framework
# MAGIC 
# MAGIC 1. Import data
# MAGIC 2. Data cleaning & preprocessing
# MAGIC 3. Sentiment Analysis
# MAGIC 4. EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data cleaning & preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ####Import data

# COMMAND ----------

#Data size
#Video Games:       (487419, 12)
#Books:             (999913, 12)
#Home and Kitchen:  (1999999, 12)


#df1 = spark.sql("select * from default.video_games_5")
#df2 = spark.sql("select * from default.books_5_small")
#df3 = spark.sql("select * from default.home_and_kitchen_5_small")

#Add Type ID column to each dataset
#----------------------------
#Type ID

#Create another column to capture unique reviewID
#----------------------------
#reviewID.2


#Data to import
#----------------------------
#df_to_use = df1
#df_to_use = df2
#df_to_use = df3

#Dashboard data to export
#----------------------------
#Dashboard_OutputName = "tmp.df1_db_jc"
#Dashboard_OutputName = "tmp.df2_db_jc"
#Dashboard_OutputName = "tmp.df3_db_jc"

#Cleaned data to export
#----------------------------
#Cleaned_OutputName = "tmp.df1_cleaned_jc"
#Cleaned_OutputName = "tmp.df2_cleaned_jc"
#Cleaned_OutputName = "tmp.df3_cleaned_jc"

# COMMAND ----------

#Rows and Columns of each file
#print((df1.count(), len(df1.columns)))
#print((df2.count(), len(df2.columns)))
#print((df3.count(), len(df3.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data cleaning & preprocessing

# COMMAND ----------

#----------------------------------------------------------------------------------
#Convert Unix timestamp to readable date
#----------------------------------------------------------------------------------


#from pyspark.sql.functions import from_unixtime, to_date

#df_to_use = df_to_use.withColumn("reviewTime", to_date(from_unixtime(df_to_use.unixReviewTime))) \
#                                                .drop("unixReviewTime")

#----------------------------------------------------------------------------------
#Fill in the empty vote column with 0, and convert it to numeric type
#----------------------------------------------------------------------------------

#from pyspark.sql.types import *

#df_fill_vote = df_to_use.withColumn("vote", df_to_use.vote.cast(IntegerType())) \
                                                 .fillna(0, subset=["vote"]) \


#----------------------------------------------------------------------------------
# Add sentiment column
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# Removing & Replace un-wanted characters (regexp_replace function is expensive)
#----------------------------------------------------------------------------------

#Remove
#df_wordCount.withColumn("RemovedSpecialCharacters",regexp_replace(col("reviewWordCleaned"), "/[^0-9]+/", ""))

#Replace


#----------------------------------------------------------------------------------
# Case Normalization (cannot use on arrays)
#----------------------------------------------------------------------------------

#from pyspark.sql.functions import lower, col

#df_lowercase = df_stop_word_removed.select("reviewWordFiltered", lower(col('reviewWordFiltered')))
#df_lowercase = df_fill_vote.withColumn("reviewText", lower(col("reviewText")))

#----------------------------------------------------------------------------------
# Tokenization
#----------------------------------------------------------------------------------

#from pyspark.ml.feature import RegexTokenizer

#regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="reviewWord", pattern="\\W")

#df_tokenized = regexTokenizer.transform(df_lowercase.fillna("", subset=["reviewText"]))

#----------------------------------------------------------------------------------
# Remove stop words
#----------------------------------------------------------------------------------

#from pyspark.ml.feature import StopWordsRemover

#remover = StopWordsRemover(inputCol="reviewWord", outputCol="reviewWordFiltered")
#df_stop_word_removed = remover.transform(df_tokenized)



#----------------------------------------------------------------------------------
# Stemming
#----------------------------------------------------------------------------------

#from nltk.stem.porter import PorterStemmer
#from pyspark.sql.functions import udf

#def stemming(col):
#  p_stemmer = PorterStemmer()
#  return [p_stemmer.stem(w) for w in col]

#stemming_udf = udf(stemming, ArrayType(StringType()))
#df_stemmed = df_stop_word_removed.withColumn("reviewWordCleaned", stemming_udf(df_stop_word_removed.reviewWordFiltered))

#reviewWordFiltered contains words prior to stemming, and reviewWordCleaned contains words after stemming


#----------------------------------------------------------------------------------
# Add a column for word count
#----------------------------------------------------------------------------------
#import pyspark.sql.functions as f

#df_wordCount = df_stemmed.withColumn('wordCount', f.size(f.split(f.col('reviewText'), ' ')))


#----------------------------------------------------------------------------------
# Add a column for count of exclaimation mark
#----------------------------------------------------------------------------------





#----------------------------------------------------------------------------------
# Text Normalization
#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ####Display Cleaned Data

# COMMAND ----------

#final_df = df_wordCount
#final_text = 'reviewWordCleaned'

#CountVectorizer
#minDF specifies in how many rows does a word need to appear for it to be counted, used for removing terms that appear too infrequently
#from pyspark.ml.feature import CountVectorizer
#cv = CountVectorizer(inputCol=final_text, outputCol="rawFeatures",minDF = 10)
#cv_model = cv.fit(final_df)
#final_df = cv_model.transform(final_df)


# COMMAND ----------

#Apply the IDF part of TF-IDF (term frequency–inverse document frequency)

#from pyspark.ml.feature import  IDF

#IDF down-weighs features which appear frequently in a corpus. This generally improves performance when using text as features since most frequent, and hence less important words, get down-weighed.
#idf = IDF(inputCol="rawFeatures", outputCol="features")
#idfModel = idf.fit(final_df)
#final_df = idfModel.transform(final_df)


# COMMAND ----------

#Drop un-needed columns
#df_cleaned = final_df.drop("reviewWord").drop("reviewWordFiltered").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform Data for Dashboard

# COMMAND ----------


#final_text_copy = final_text+"_copy"


#Convert array into strings
#from pyspark.sql.functions import col, concat_ws

#df_dashboard = df_cleaned.withColumn(final_text_copy,
#   concat_ws(" ",col(final_text)))

#Create data frame with all words in one column

#df_dashboard = df_dashboard.withColumn('word', f.explode(f.split(f.col(final_text_copy), ' ')))
#df_dashboard = df_dashboard\
#.drop("reviewText")\
#.drop("features")\
#.drop("summary")\
#.drop(final_text)\
#.drop(final_text_copy)\
#.cache()


#display(df_dashboard)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Export into tmp folder

# COMMAND ----------

#df_cleaned.write.format("parquet").saveAsTable('tmp.df_cleaned_jc')
#df_dashboard.write.format("parquet").mode("overwrite").saveAsTable(Dashboard_OutputName)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ---------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #### Building a Pipeline

# COMMAND ----------

#from pyspark.ml import Pipeline
#from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# We'll tokenize the text using a simple RegexTokenizer
#regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")


# Remove standard Stopwords
#stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")


# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
#countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)


#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Running the Pipeline

# COMMAND ----------

#pipelineFit = pipeline.fit(df1)
#df1_cleaned = pipelineFit.transform(df1)
#display(df1_cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sentiment analysis (ML-based)

# COMMAND ----------

#Steps: 

#1. We will use the following crtieria to create a sentiment label:
# overall — rating of the product 4-5 = 1 Positive
# overall — rating of the product 1-2 = 0 Negative

# COMMAND ----------

#Select a df that we will use for modelling
#Neutral ratings (=3) are removed

#df_sentiment = df_cleaned.drop("reviewID")\
#.drop("verified")\
#.drop("reviewerID")\
#.drop("reviwerName")\
#.drop("reviewText")\
#.drop("unixReviewTime")\
#.drop("words")\
#.drop("wordCount")\
#.drop("rawFeatures")\
#.filter("overall !=3")


# COMMAND ----------

#Group the overall score into sentiment labels

#from pyspark.ml.feature import Bucketizer

#splits = [-float("inf"), 4.0, float("inf")]
#bucketizer = Bucketizer(splits=splits, inputCol="overall", outputCol="label_overall")

#df_sentimentbucket = bucketizer.transform(df_sentiment)

#Show results and count of split
#df_sentimentbucket.groupBy("overall","label_overall").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check data imbalance

# COMMAND ----------

#Check target imbalance

#dataset_size=float(df_sentimentbucket.select("label_overall").count())
#numPositives=df_sentimentbucket.select("label_overall").where('label_overall == 1').count()
#per_ones=(float(numPositives)/float(dataset_size))*100
#numNegatives=float(dataset_size-numPositives)
#print('The number of 1s are {}'.format(numPositives))
#print('Percentage of 1s are {}'.format(per_ones))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stratified sampling (Undersampling)

# COMMAND ----------

#This step is to make sure we have a balanced dataset

#Check existing imbalance

#from pyspark.sql.functions import lower, col

#major_df = df_sentimentbucket.filter(col("label_overall") == 1)
#minor_df = df_sentimentbucket.filter(col("label_overall") == 0)
#ratio = int(major_df.count()/minor_df.count())
#print("ratio: {}".format(ratio))


# COMMAND ----------

#Perform undersampling because of large data size

#sampled_majority_df = major_df.sample(False, 1/ratio, seed=1234)
#df_sentimentbucket_balance = sampled_majority_df.unionAll(minor_df)

#Check updated balance
#new_major_df = df_sentimentbucket_balance.filter(col("label_overall") == 1)
#new_minor_df = df_sentimentbucket_balance.filter(col("label_overall") == 0)
#new_ratio = int(new_major_df.count()/new_minor_df.count())
#print("ratio: {}".format(new_ratio))

#Check by overall score
#df_sentimentbucket_balance.groupBy("overall","label_overall").count().show()
#Make sure ratio = 1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split data to train/test set

# COMMAND ----------

#split into training and test dataset
#train_sentiment, test_sentiment = df_sentimentbucket_balance.randomSplit([0.8, 0.2], seed=12345)

#Check imbalance in train set
#train_dataset_size=float(train_sentiment.select("label_overall").count())
#train_numPositives=train_sentiment.select("label_overall").where('label_overall == 1').count()
#train_per_ones=(float(train_numPositives)/float(train_dataset_size))*100
#train_numNegatives=float(train_dataset_size-train_numPositives)
#print('The number of 1s in train are {}'.format(train_numPositives))
#print('Percentage of 1s in train are {}'.format(train_per_ones))

#Check imbalance in test set
#test_dataset_size=float(test_sentiment.select("label_overall").count())
#test_numPositives=test_sentiment.select("label_overall").where('label_overall == 1').count()
#test_per_ones=(float(test_numPositives)/float(test_dataset_size))*100
#test_numNegatives=float(test_dataset_size-test_numPositives)
#print('The number of 1s in test are {}'.format(test_numPositives))
#print('Percentage of 1s in test are {}'.format(test_per_ones))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression (with fit elastic net regularization)

# COMMAND ----------

#Train Model
#from pyspark.ml.classification import LogisticRegression

#lambda_par = 0.02
#alpha_par = 0.3

#lr = LogisticRegression(labelCol="label_overall", featuresCol="features",maxIter=20, regParam=lambda_par, elasticNetParam=alpha_par)
#lr_model=lr.fit(train_sentiment)

#predict_train=lr_model.transform(train_sentiment)

# COMMAND ----------

#Estimating accuracy

#from pyspark.sql import functions as fn

#lr_model.transform(test_sentiment).\
#    select(fn.expr('float(prediction = label_overall)').alias('correct')).\
#    select(fn.avg('correct')).show()

#Shows 0.862355044
#Shows 0.834532704

# COMMAND ----------

#Create a table of words with their weights

#import pandas as pd
#vocabulary = cv_model.vocabulary
#weights = lr_model.coefficients.toArray()
#coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})


# COMMAND ----------

#Most negative words
#df_negative_sentiment = coeffs_df.sort_values('weight', ascending=False).head(50)

#display(df_negative_sentiment)

# COMMAND ----------

#Most positive words
#df_positive_sentiment = coeffs_df.sort_values('weight').head(50)

#display(df_positive_sentiment)

# COMMAND ----------

#Append two sentiment dataset
#df_sentiment = df_positive_sentiment.append(df_negative_sentiment, ignore_index=True)

#display(df_sentiment)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Parameter tunning

# COMMAND ----------

#Parameter tunning to find optimal lambda_par and alpha_par

#from pyspark.ml.tuning import ParamGridBuilder

#grid = ParamGridBuilder().\
#    addGrid(lr_model.regParam, [0., 0.01, 0.02]).\
#    addGrid(lr_model.elasticNetParam, [0., 0.2, 0.4]).\
#    build()

#all_models = []
#for j in range(len(grid)):
#    print("Fitting model {}".format(j+1))
#    reg_model = lr.fit(train_sentiment, grid[j])
#    all_models.append(reg_model)

# COMMAND ----------

#accuracies = [m.\
#    transform(test_sentiment).\
#    select(fn.avg(fn.expr('float(label_overall = prediction)')).alias('accuracy')).\
#    first().\
#    accuracy for m in all_models]



# COMMAND ----------

accuracies

# COMMAND ----------

#Test model

#predict_test=lr_model.transform(test_sentiment)
#predict_test.select("label_overall","prediction").show(20)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform Data for Dashboard

# COMMAND ----------


#final_text_copy = final_text+"_copy"


#Convert array into strings
#from pyspark.sql.functions import col, concat_ws

#df_dashboard = df_cleaned.withColumn(final_text_copy,
#   concat_ws(" ",col(final_text)))

#Create data frame with all words in one column

#df_dashboard = df_dashboard.withColumn('word', f.explode(f.split(f.col(final_text_copy), ' ')))
#df_dashboard = df_dashboard\
#.drop("reviewText")\
#.drop("features")\
#.drop("summary")\
#.drop(final_text)\
#.drop(final_text_copy)\
#.cache()


#display(df_dashboard)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory analysis

# COMMAND ----------

#import pyspark.sql.functions as f

#Count the number of words in the reviewText column
#df1_wordrowcount = final_df.withColumn('wordCount', f.size(f.split(f.col('reviewText'), ' ')))

#Plot average word count grouped by vote
#display(df1_wordrowcount.groupBy("vote").avg("wordCount").orderBy("vote"))

# COMMAND ----------

#import pyspark.sql.functions as f

#Plot top 20 words
#top20words = df2.withColumn('word', f.explode(f.split(f.col(cleaned_reviewText), ' ')))\
#    .groupBy('word')\
#    .count()\
#    .sort('count', ascending=False)\
#    .limit(20)

#display(top20words)

# COMMAND ----------

#from pyspark.sql.window import Window
#from pyspark.sql.functions import rank, col

#Create a table of word frequency grouped by overall
#df_wordfrequency = df2.withColumn('word', f.explode(f.split(f.col(cleaned_reviewText), ' ')))\
#    .groupBy('word','overall')\
#    .count()

