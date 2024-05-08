import os
import sys
from dataclasses import dataclass

import polars as pl

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging

TRAIN_TEST_SPLIT = 0.2


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")
            df = pl.read_csv('notebook/data/stud.csv')
            logging.info("Data ingestion completed")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.write_csv(self.ingestion_config.raw_data_path)

            logging.info(
                f"Train/Test split initiated with split ratio of {int(TRAIN_TEST_SPLIT*100)}%"
            )
            df = df.sample(n=len(df), shuffle=True)
            split_index = int(len(df) * TRAIN_TEST_SPLIT)
            test_df = df.slice(0, split_index)
            train_df = df.slice(split_index, len(df))

            train_df.write_csv(self.ingestion_config.train_data_path)
            test_df.write_csv(self.ingestion_config.test_data_path)

            logging.info("Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
