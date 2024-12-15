import os
import sys
import logging
from dotenv import load_dotenv
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Fraud Detection Preprocessing") \
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
    .config("spark.local.dir", "/tmp") \
    .getOrCreate()


def preprocess(csv_file_path, output_file_path):
    """
    Preprocesses a CSV file for fraud detection.
    Args:
        csv_file_path (str): Path to the input CSV file.
        output_file_path (str): Path to save the output Parquet file.
    Returns:
        DataFrame: Preprocessed Spark DataFrame.
    """
    try:
        logging.info("Preprocessing Pipeline: Started running pipeline for data pre-processing.")

        # Read the local CSV file into a Spark DataFrame
        logging.info(f"Reading CSV file from {csv_file_path}.")
        spark_df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

        # Check if the necessary columns are present
        required_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
        ]
        missing_columns = [col for col in required_columns if col not in spark_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the dataset: {missing_columns}")

        # Remove duplicates
        logging.info("Removing duplicate rows.")
        spark_df = spark_df.distinct()

        # Assemble feature vector
        logging.info("Assembling feature vector.")
        numericCols = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]
        assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
        spark_df = assembler.transform(spark_df).select("features", "Class")

        # Standardize input feature vector
        logging.info("Standardizing feature vector.")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
        scalerModel = scaler.fit(spark_df)
        spark_df = scalerModel.transform(spark_df)

        # Save the pre-processed data as a Parquet file locally
        logging.info(f"Saving preprocessed data to {output_file_path}.")
        spark_df.write.mode("overwrite").parquet(output_file_path)

        logging.info("Preprocessing Pipeline: Completed successfully.")
        return spark_df

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Check for required command-line arguments
        if len(sys.argv) != 3:
            raise IndexError("Please provide both the CSV file path and the output path as arguments.")

        # Get the input CSV file path and output path from command-line arguments
        csv_file_path = sys.argv[1]  # Local CSV file path
        output_file_path = sys.argv[2]  # Path to save the output Parquet file

        # Run preprocessing
        preprocess(csv_file_path, output_file_path)

    except IndexError:
        logging.error("Please provide both the CSV file path and the output path as arguments.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
