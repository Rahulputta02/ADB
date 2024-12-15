import os
import sys
import shutil
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from dotenv import load_dotenv
from pyspark.sql import SparkSession

load_dotenv()

# Local Spark session setup with specific configurations to bypass Hadoop security issues
spark = SparkSession.builder \
    .appName("Fraud Detection Training") \
    .master("local[*]") \
    .config("spark.hadoop.security.authentication", "none") \
    .getOrCreate()  # Ensure the SparkSession is correctly created

def train_model(df):
    # Split the data into training and test sets
    train, test = df.randomSplit([0.7, 0.3], seed=2018)

    # Define the random forest classifier
    rf_clf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='Class')
    rfModel = rf_clf.fit(train)

    # Make predictions on the test set
    predictions = rfModel.transform(test)

    # Evaluate the model's accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)

    print(f"Training Pipeline: Model training completed. Overall accuracy is {accuracy:.4f}")

    # Save the trained model
    model_path = "model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)  # Clean up existing model folder
    rfModel.save(model_path)

    return accuracy

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Error: Please provide the path to the preprocessed parquet file as an argument.")
        sys.exit(1)

    # Get the parquet file path
    parquet_file_path = sys.argv[1]

    print(f"Reading parquet file from: {parquet_file_path}")

    # Ensure the input parquet file exists
    if not os.path.exists(parquet_file_path):
        print(f"Error: The specified parquet file {parquet_file_path} does not exist.")
        sys.exit(1)

    # Read the preprocessed data from parquet file
    try:
        preprocessed_df = spark.read.parquet(parquet_file_path)
        print(f"Successfully read the parquet file with {preprocessed_df.count()} records.")
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    # Train the model
    accuracy = train_model(preprocessed_df)

    # Optionally, output accuracy or log it
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
