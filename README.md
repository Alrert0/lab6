# Lab 6: Spark ML Pipeline on Amazon EMR

## Project Overview
This project demonstrates an end-to-end distributed machine learning pipeline using Apache Spark on an Amazon EMR cluster to predict bank customer churn. The pipeline performs distributed feature engineering, model training, and evaluation at scale using PySpark ML.

## Lab Objectives
- Build an end-to-end Spark ML pipeline
- Perform distributed feature engineering using Spark DataFrame operations and Transformers
- Train and evaluate classification models in a distributed environment
- Monitor job execution using YARN and the Spark UI

## Platform & Tools
- Platform: Amazon EMR (1 Master node, 2 Core nodes)
- Instance Types: m4.large
- Frameworks: Apache Spark (PySpark), Hadoop (HDFS), YARN
- Languages: Python (PySpark)
- Dataset: Bank Customer Churn Dataset (Kaggle)

## Dataset
The dataset contains bank customer information with features such as:
- CreditScore, Geography, Gender, Age, Tenure, Balance
- NumOfProducts, EstimatedSalary, HasCrCard, IsActiveMember, etc.

Target variable:
- `Exited` (1 = Churn, 0 = Stayed)

## Pipeline Stages
1. Data Loading: Read CSV from HDFS into a Spark DataFrame
2. Categorical Encoding: `StringIndexer` + `OneHotEncoder` for `Geography` and `Gender`
3. Feature Assembly: `VectorAssembler` to combine numeric and encoded features
4. Feature Scaling: `StandardScaler` to normalize the assembled feature vector
5. Model Training: Train classification models (Logistic Regression, Random Forest) using `Pipeline` and `CrossValidator`/`TrainValidationSplit` as needed
6. Evaluation: Assess performance using `MulticlassClassificationEvaluator` (Accuracy) and other metrics if desired

## How to run on an EMR cluster
1. Upload the dataset to the Master node:

```bash
scp -i labsuser.pem Churn_Modelling.csv hadoop@<master-public-dns>:/home/hadoop/
```

2. Prepare HDFS directories and upload the data:

```bash
hdfs dfs -mkdir -p /user/hadoop/churn_input
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/
```

3. (Optional) Install missing Python dependencies on EMR master (use with caution on managed clusters):

```bash
sudo pip3 install numpy
```

4. Submit the Spark job:

```bash
spark-submit --master yarn --deploy-mode client churn_pipeline.py
```

Notes:
- Replace `churn_pipeline.py` with your script path. On EMR, you can also use the EMR Steps API or the console to submit jobs.
- If using S3 for input/output, update the job to read/write from `s3://<bucket>/path` and ensure the EMR role has S3 permissions.

## Model Comparison (Required Experiment: Option C)
I compared two classification algorithms within the Spark ML Pipeline:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.8069 |
| Random Forest | 0.8553 |

Observation: The Random Forest model outperformed Logistic Regression by ~4.8 percentage points, indicating non-linear ensemble methods better capture churn patterns in this dataset.

## Monitoring
- Monitored the job via the YARN Resource Manager and Spark UI
- Application Name: `CustomerChurnComparison`
- State: FINISHED
- Final Status: SUCCEEDED

On EMR, you can access the YARN Resource Manager UI and Spark History Server through the EMR console or by setting up an SSH tunnel to the master node.

## Files in this repository (suggested)
- `churn_pipeline.py` - main PySpark script implementing the pipeline
- `requirements.txt` - Python dependencies (if needed)
- `data/` - optional local sample data for testing

## Conclusion
This lab showcases how Spark ML on Amazon EMR can be used to build scalable, distributed machine learning pipelines. The EMR environment enables parallel feature engineering and model training, and the Random Forest classifier showed improved performance over Logistic Regression on this churn prediction task.

