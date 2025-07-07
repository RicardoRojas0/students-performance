import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    """
    Saves object in specific file path
    """
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)
        with open(file=file_path, mode="wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)


def load_object(file_path):
    """
    Load object from specific path
    """
    try:
        with open(file=file_path, mode="rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report[name] = r2_score(y_true=y_test, y_pred=y_pred)
        return report
    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)
