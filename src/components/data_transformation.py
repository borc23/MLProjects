import os
import sys
from dataclasses import dataclass

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    This code is responsible for transforming the data. It uses the preprocessor object to transform the data.
    """

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Categorical columns encoding completed")

            logging.info("Numerical columns: %s", numerical_columns)
            logging.info("Categorical columns: %s", categorical_columns)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipelines", numerical_pipeline, numerical_columns),
                    ("cat_pipelines", categorical_pipeline, categorical_columns),
                ]
            )
            logging.info("Column transformation completed")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pl.read_csv(train_data_path)
            test_df = pl.read_csv(test_data_path)

            logging.info("Read train and test data successfully")

            logging.info("Opening preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = train_df.select(cs.numeric()).select(pl.exclude(target_column_name)).columns
            

            input_features_df_train = train_df.drop(columns=[target_column_name])
            target_feature_df_train = train_df.select(target_column_name)

            input_features_df_test = test_df.drop(columns=[target_column_name])
            target_feature_df_test = test_df.select(target_column_name)

            input_features_df_train = input_features_df_train.to_pandas(use_pyarrow_extension_array=True)
            target_feature_df_train = target_feature_df_train.to_pandas(use_pyarrow_extension_array=True)

            input_features_df_test = input_features_df_test.to_pandas(use_pyarrow_extension_array=True)
            target_feature_df_test = target_feature_df_test.to_pandas(use_pyarrow_extension_array=True)

            logging.info("Applying preprocessing object on data")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_features_df_train
            )
            input_feature_test_arr = preprocessing_obj.transform(input_features_df_test)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_df_train)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_df_test)]

            logging.info("Data transformation completed successfully")

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
