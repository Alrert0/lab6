from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Создание сессии
spark = SparkSession.builder.appName("CustomerChurnComparison").getOrCreate()

# 2. Загрузка данных
data = spark.read.csv("hdfs:///user/hadoop/churn_input/Churn_Modelling.csv", header=True, inferSchema=True)

# 3. Подготовка признаков (Feature Engineering)
geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

assembler = VectorAssembler(
    inputCols=["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "GeographyVec", "GenderVec"],
    outputCol="features"
)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Разделение данных на обучение и тест
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# --- МОДЕЛЬ 1: Logistic Regression ---
lr = LogisticRegression(labelCol="Exited", featuresCol="scaledFeatures")
pipeline_lr = Pipeline(stages=[geo_indexer, gender_indexer, encoder, assembler, scaler, lr])
model_lr = pipeline_lr.fit(train_data)
predictions_lr = model_lr.transform(test_data)

# --- МОДЕЛЬ 2: Random Forest (Эксперимент) ---
rf = RandomForestClassifier(labelCol="Exited", featuresCol="scaledFeatures", numTrees=10)
pipeline_rf = Pipeline(stages=[geo_indexer, gender_indexer, encoder, assembler, scaler, rf])
model_rf = pipeline_rf.fit(train_data)
predictions_rf = model_rf.transform(test_data)

# 4. Оценка результатов
evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction", metricName="accuracy")

accuracy_lr = evaluator.evaluate(predictions_lr)
accuracy_rf = evaluator.evaluate(predictions_rf)

print("\n" + "="*30)
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print("="*30 + "\n")

# Показать пример предсказаний
predictions_rf.select("CustomerId", "Surname", "Exited", "prediction").show(10)

spark.stop()