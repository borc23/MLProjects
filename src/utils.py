import os
import sys

import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        model_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models)[i]
            logging.info(f"Training {model_name} model")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[list(models.keys())[i]] = r2

        return model_report
    except Exception as e:
        raise CustomException(e, sys)