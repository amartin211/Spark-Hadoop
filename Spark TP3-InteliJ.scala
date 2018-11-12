// Databricks notebook source
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.Interaction
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext, sql}
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import spark.implicits._

val my_data7= sqlContext.read.parquet("/FileStore/tables/part_00007_da4490f3_3af7_4783_a4a9_64355a59cb83_c000_snappy-6155f.parquet")


// COMMAND ----------

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

// COMMAND ----------

val tokenizer = new RegexTokenizer()
  .setPattern("\\W+")
  .setGaps(true)
  .setInputCol("text")
  .setOutputCol("tokens")


// COMMAND ----------

val remover = new StopWordsRemover()
 .setInputCol("tokens")
 .setOutputCol("filtered")


// COMMAND ----------

val cvModel = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("vector_features")
 


// COMMAND ----------



val idf = new IDF()
.setInputCol("vector_features")
.setOutputCol("tfidf")




// COMMAND ----------

val pipeline = new Pipeline()
  .setStages(Array(tokenizer, remover, cvModel, idf))

val model =pipeline.fit(my_data7)
    val df = model.transform(my_data7)
    println(df.show())

// COMMAND ----------

val indexer1 = new StringIndexer()
  .setInputCol("country2")
  .setOutputCol("country_indexed")
 


val indexer2 = new StringIndexer()
  .setInputCol("currency2")
  .setOutputCol("currency_indexed")



val encoder = new OneHotEncoderEstimator()
  .setInputCols(Array("country_indexed", "currency_indexed"))
  .setOutputCols(Array("country_onehot","currency_onehot" ))





// COMMAND ----------

 val pipeline2 = new Pipeline()
      .setStages(Array(indexer1, indexer2,encoder))

    val model2 =pipeline2.fit(df)
    val df2 = model2.transform(df)
    println(df2.show())

// COMMAND ----------

val assembler = new VectorAssembler()
  .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
  .setOutputCol("features")

// COMMAND ----------

val lr = new LogisticRegression()
   .setElasticNetParam(0.0)   
   .setFitIntercept(true)
   .setFeaturesCol("features")
   .setLabelCol("final_status")
   .setStandardization(true)
   .setPredictionCol("predictions")
   .setRawPredictionCol("raw_predictions")
   .setThresholds(Array(0.7, 0.3))
   .setTol(1.0e-6)
   .setMaxIter(300)

// COMMAND ----------

val pipeline_final = new Pipeline()
  .setStages(Array(tokenizer, remover, cvModel, idf, indexer1, indexer2, encoder, assembler, lr))

// COMMAND ----------

 val final_model = pipeline_final.fit(my_data7)
    val final_df = final_model.transform(my_data7)
    println(final_df.show())

// COMMAND ----------

val Array(training, test) = my_data7.randomSplit(Array(0.9, 0.1), seed = 12345)

// COMMAND ----------

val paramGrid = new ParamGridBuilder()
  .addGrid(cvModel.minDF, Array(55.toDouble, 75.toDouble, 95.toDouble))
  .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4,10e-2))
  .build()



// COMMAND ----------

val f1_score = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

// COMMAND ----------

val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(pipeline_final)
  .setEvaluator(f1_score)
  .setEstimatorParamMaps(paramGrid)
  // 80% of the data will be used for training and the remaining 20% for validation.
  .setTrainRatio(0.7)
  // Evaluate up to 2 parameter settings in parallel
  

// Run train validation split, and choose the best set of parameters.


// COMMAND ----------

val model_final = trainValidationSplit.fit(training)

// COMMAND ----------

val df_WithPredictions = model_final.transform(test)
val score_final = f1_score.evaluate(df_WithPredictions)
println("f1 score: "+ score_final)




// COMMAND ----------


df_WithPredictions.groupBy("final_status", "predictions").count.show()


// COMMAND ----------

}
}
