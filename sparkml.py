from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session with Hive support
spark = SparkSession.builder.appName("EmployeePerformanceRegression").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'employee_performance'
df = spark.sql("SELECT Experience_Years, Monthly_Sales FROM employee_performance")

# Step 3: Handle null values by dropping rows with missing data
df = df.na.drop()

# Step 4: Prepare data for MLlib (feature vector assembly)
assembler = VectorAssembler(
    inputCols=["Experience_Years"],
    outputCol="features",
    handleInvalid="skip"
)
assembled_df = assembler.transform(df).select("features", "Monthly_Sales")

# Step 5: Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Step 6: Initialize and train a Linear Regression model
lr = LinearRegression(labelCol="Monthly_Sales")
lr_model = lr.fit(train_data)

# Step 7: Evaluate the model on the test data
test_results = lr_model.evaluate(test_data)

# Step 8: Print the model performance metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase ----
data = [
    ('metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'cf:r2',   str(test_results.r2)),
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')  # Update if your HBase host differs
    connection.open()
    table = connection.table('employee_metrics')  # <-- update table name as needed
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop Spark session
spark.stop()
