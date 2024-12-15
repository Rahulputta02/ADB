import os
import sys
import shutil
import logging
import uuid
import pandas as pd
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Spark session
def get_spark():
    return SparkSession.builder \
        .appName("Fraud Detection") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .getOrCreate()


def preprocess(input_file_path):
    """
    Preprocess input data for training or inference.
    Args:
        input_file_path (str): Path to the input CSV file.
    Returns:
        Spark DataFrame: Preprocessed data ready for training or prediction.
    """
    try:
        logging.info("Starting pre-processing.")
        spark = get_spark()
        logging.info(f"Reading input file: {input_file_path}")

        # Load CSV file
        spark_df = spark.read.csv(input_file_path, header=True, inferSchema=True)

        # Check if the necessary columns exist
        required_cols = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
        ]
        missing_cols = [col for col in required_cols if col not in spark_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        # Remove duplicates
        spark_df = spark_df.distinct()

        # Assemble features
        logging.info("Assembling feature vector.")
        assembler = VectorAssembler(inputCols=required_cols[:-2], outputCol="features")
        spark_df = assembler.transform(spark_df).select('features', 'Class', 'Amount')

        # Standardize features
        logging.info("Standardizing feature vectors.")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
        scaler_model = scaler.fit(spark_df)
        spark_df = scaler_model.transform(spark_df)

        logging.info("Pre-processing completed successfully.")
        return spark_df

    except Exception as e:
        logging.error(f"Error during pre-processing: {e}")
        raise


def train_and_save_model(input_file_path, model_path):
    """
    Train and save a Random Forest model.
    Args:
        input_file_path (str): Path to the input CSV file for training.
        model_path (str): Path to save the trained model.
    """
    try:
        logging.info("Training model.")
        spark_df = preprocess(input_file_path)

        # Split data into training and testing sets
        train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)

        # Train Random Forest model
        rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="Class", numTrees=50)
        model = rf.fit(train_data)

        # Save the model, overwriting if necessary
        logging.info(f"Saving model to: {model_path}")
        model.write().overwrite().save(model_path)

        logging.info("Model training and saving completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise


def run_inference(test_file_path, model_path):
    """
    Run inference on a dataset using a pre-trained model.
    Args:
        test_file_path (str): Path to the test data CSV file.
        model_path (str): Path to the saved Random Forest model.
    """
    try:
        # Pre-process the data
        processed_df = preprocess(test_file_path)

        # Load the trained model
        logging.info(f"Loading model from: {model_path}")
        model = RandomForestClassificationModel.load(model_path)

        # Run predictions
        logging.info("Running predictions.")
        predictions = model.transform(processed_df)
        predictions = predictions.select("Amount", "Class", "prediction")

        # Save predictions to a CSV
        output_dir = "predictions_output"
        os.makedirs(output_dir, exist_ok=True)
        predictions.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv(output_dir)

        # Load the CSV into a Pandas DataFrame for easier handling
        output_files = os.listdir(output_dir)
        csv_file = next((f for f in output_files if f.endswith(".csv")), None)
        if not csv_file:
            raise FileNotFoundError("No predictions CSV file generated.")
        csv_path = os.path.join(output_dir, csv_file)
        prediction_data = pd.read_csv(csv_path)

        # Log predictions
        logging.info("Logging predictions.")
        for _, row in prediction_data.iterrows():
            prediction_log = {
                "prediction_id": str(uuid.uuid4()),
                "amount": row["Amount"],
                "actual_class": row["Class"],
                "predicted_class": int(row["prediction"])
            }
            logging.info(f"Prediction: {prediction_log}")

        # Clean up temporary files
        shutil.rmtree(output_dir)

        logging.info(f"Inference completed successfully for {test_file_path}.")

    except Exception as e:
        logging.error(f"Inference Pipeline Error: {e}")
        raise


if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            raise ValueError("Usage: python3 script.py <mode> <file_path> <model_path>")

        mode = sys.argv[1]  # "train" or "infer"
        file_path = sys.argv[2]
        model_path = sys.argv[3]

        if mode == "train":
            train_and_save_model(file_path, model_path)
        elif mode == "infer":
            run_inference(file_path, model_path)
        else:
            raise ValueError("Invalid mode. Use 'train' to train the model or 'infer' to run inference.")

    except Exception as e:
        logging.error(f"Script Error: {e}")
        sys.exit(1)
