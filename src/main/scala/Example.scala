import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.regression.{GBTRegressionModel, LinearRegression}
import org.apache.spark.sql.SparkSession

object Example {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val df = spark
      .read
      .option("inferSchema", true)
      .option("header", true)
      .csv("data/realestate.csv")

    df.printSchema()

    val Array(train, valid) = df.randomSplit(Array(0.7, 0.3))

    train.show()
    valid.show()

    val rForm = new RFormula()
      .setFormula("price ~ rooms")

    val trainDF = rForm.fit(train)

    val fittedRF = rForm.fit(train)
    val preparedTrain = fittedRF.transform(train)

    val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features")

    val model = lr.fit(preparedTrain)

    println(model.intercept)
    println(model.coefficients)

    val test = spark
      .read
      .option("header", true)
      .option("inferSchema", true)
      .csv("data/test.csv")

    val preparedTest = fittedRF.transform(test)

    val predictions = model.transform(preparedTest)
    predictions.show()


  }
}
