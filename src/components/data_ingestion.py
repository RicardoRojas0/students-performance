import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


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
        logging.info("Initiating data ingestion method")
        try:
            df = pd.read_csv(
                "/Users/ricardo-rojas/Documents/GitHub/students-performance/data/students_performance.csv"
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
            logging.info("Data ingestion completed")

            return (
                self.data_ingestion_config.raw_data_path,
                self.data_ingestion_config.ingested_train_path,
                self.data_ingestion_config.ingested_test_path,
            )
        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.data_ingestion()
