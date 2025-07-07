import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _data_preprocessor(self):
        """
        Function responsible for data transformation.
        """
        try:
            numerical_features = [
                "writing_score",
                "reading_score",
            ]

            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info("Numerical features imputed and scaled")

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Categorical features imputed, encoded and scaled")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_features,
                    ),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def data_transformation(self, ingested_train_path, ingested_test_path):
        try:
            logging.info("--- INITIATING DATA TRANSFORMATION PROCESS ---")

            df_train = pd.read_csv(ingested_train_path)
            df_test = pd.read_csv(ingested_test_path)
            logging.info("Read train and test sets.")

            logging.info("Obtaining preprocessor")
            preprocessor = self._data_preprocessor()
            target_col = "math_score"

            logging.info("Initiating data split process")
            # Splittin data into train and test sets, and into features and target sets
            df_train_features = df_train.drop(columns=[target_col])
            df_train_target = df_train[target_col]

            df_test_features = df_test.drop(columns=[target_col])
            df_test_target = df_test[target_col]
            logging.info("Split data process completed")

            logging.info("Applying preprocessor to training and test")
            df_train_processed = preprocessor.fit_transform(df_train_features)
            df_test_processed = preprocessor.transform(df_test_features)

            # Concatenate feature arrays with target arrays for both datasets
            train_array = np.c_[df_train_processed, np.array(df_train_target)]
            test_array = np.c_[df_test_processed, np.array(df_test_target)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor,
            )
            logging.info("Preprocessor objects successfully saved")
            logging.info("--- DATA TRANSFORMATION PROCESS COMPLETED ---")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
