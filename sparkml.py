from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("EmployeePerformance_to_HBase").enableHiveSupport().getOrCreate()

# Step 2: Load data from Hive and cast to INT (prevents IllegalArgumentException)
df = spark.sql("""
  SELECT
    CAST(Experience_Years AS INT) AS Experience_Years,
    CAST(Monthly_Sales AS INT) AS Monthly_Sales
  FROM employee_performance
""")

# Step 3: Drop any rows with nulls
df = df.na.drop()

# Step 4: Assemble feature vector
assembler = VectorAssembler(
    inputCols=["Experience_Years"],
    outputCol="features",
    handleInvalid="skip"
)
assembled_df = assembler.transform(df).select("features", "Monthly_Sales")

# Step 5: Split into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Step 6: Train Linear Regression model
lr = LinearRegression(labelCol="Monthly_Sales")
lr_model = lr.fit(train_data)

# Step 7: Evaluate model
test_results = lr_model.evaluate(test_data)

# Step 8: Print metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# Step 9: Write metrics to HBase
data = [
    ('metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'cf:r2', str(test_results.r2)),
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')  # Update if needed
    connection.open()
    table = connection.table('employee_metrics')  # Must exist with column family 'cf'
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 10: Stop Spark
spark.stop()
