from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Start Spark session with Hive support
spark = SparkSession.builder.appName("EmployeePerformanceRegression").enableHiveSupport().getOrCreate()

# Step 2: Load from Hive (all strings originally)
df = spark.sql("SELECT Experience_Years, Monthly_Sales FROM employee_performance")

# Step 3: Cast columns to integer for regression
df = df.withColumn("Experience_Years", col("Experience_Years").cast("int")) \
       .withColumn("Monthly_Sales", col("Monthly_Sales").cast("int")) \
       .na.drop()

# Step 4: Assemble features into vector
assembler = VectorAssembler(
    inputCols=["Experience_Years"],
    outputCol="features",
    handleInvalid="skip"
)
assembled_df = assembler.transform(df).select("features", "Monthly_Sales")

# Step 5: Split into train/test
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Step 6: Train linear regression model
lr = LinearRegression(labelCol="Monthly_Sales")
lr_model = lr.fit(train_data)

# Step 7: Evaluate model
test_results = lr_model.evaluate(test_data)
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# Step 8: Write metrics to HBase
data = [
    ('metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'cf:r2', str(test_results.r2))
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')  # Update host if needed
    connection.open()
    table = connection.table('employee_metrics')
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop Spark
spark.stop()
