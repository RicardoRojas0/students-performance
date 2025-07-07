import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info("--- INITIATING MODEL TRAINIG PROCESS ---")
            logging.info("Splitting train and test sets into features and target sets.")
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(random_state=42),
                "Ridge": Ridge(random_state=42),
                "K-Neighbors": KNeighborsRegressor(n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_jobs=-1, random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "CatBoost": CatBoostRegressor(
                    random_seed=42, thread_count=-1, verbose=False
                ),
                "XGBoost": XGBRegressor(),
            }

            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            best_model_name, best_model_score = max(
                model_report.items(), key=lambda item: item[1]
            )

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found. No model got higher that 0.6."
                )

            logging.info(
                f"Best found model: {best_model_name} with an R2 Score of: {best_model_score}"
            )

            best_model = models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.model_file_path, obj=best_model
            )

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_true=y_test, y_pred=y_pred)

            logging.info("--- MODEL TRAINING PROCESS COMPLETED ---")
            return r2

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
