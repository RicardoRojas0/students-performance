import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# Using dataclass since we are only defining variables in the class
@dataclass
class DataIngestionConfig:
    ingested_train_path: str = os.path.join("artifacts", "train.csv")
    ingested_test_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def data_ingestion(self):
        logging.info("--- INITIATING DATA INGESTION PROCESS ---")
        try:
            df = pd.read_csv(
                "data/students_performance.csv"
            )
            logging.info("Read dataset as DataFrame")

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.raw_data_path),
                exist_ok=True,
            )

            # Saving raw data into specific path
            df.to_csv(
                self.data_ingestion_config.raw_data_path,
                index=False,
                header=True,
            )

            logging.info("Intiating train test split")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
            )

            # Saving traing and test set into specific path
            train_set.to_csv(
                self.data_ingestion_config.ingested_train_path,
                index=False,
                header=True,
            )

            test_set.to_csv(
                self.data_ingestion_config.ingested_test_path,
                index=False,
                header=True,
            )
            logging.info("Raw data, train and test sets saved in artifacts folder")
            logging.info("--- DATA INGESTION PROCESS COMPLETED ---")

            return (
                self.data_ingestion_config.raw_data_path,
                self.data_ingestion_config.ingested_train_path,
                self.data_ingestion_config.ingested_test_path,
            )
        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    _, ingested_train_path, ingested_test_path = data_ingestion.data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.data_transformation(
        ingested_train_path=ingested_train_path,
        ingested_test_path=ingested_test_path,
    )

    model_trainer = ModelTrainer()
    score = model_trainer.model_trainer(train_array=train_array, test_array=test_array)
    print(score)
