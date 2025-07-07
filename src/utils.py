import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException


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
