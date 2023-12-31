from collections import namedtuple
from datetime import datetime
import uuid
from customer_segment.config.configuration import Configuration
from customer_segment.logger import logging
from customer_segment.exception import CustomException
from threading import Thread
from typing import List

from multiprocessing import Process
from customer_segment.entity.artifact_entity import DataIngestionArtifact
from customer_segment.entity.artifact_entity import DataValidationArtifact
from customer_segment.entity.artifact_entity import DataTransformationArtifact
from customer_segment.entity.artifact_entity import ModelTrainerArtifact
from customer_segment.entity.artifact_entity import ModelEvaluationArtifact
from customer_segment.entity.artifact_entity import ModelPusherArtifact
from customer_segment.components.data_ingestion import DataIngestion
from customer_segment.components.data_validation import DataValidation
from customer_segment.components.data_transformation import DataTransformation
from customer_segment.components.model_trainer import ModelTrainer
from customer_segment.components.model_evaluation import ModelEvaluation
from customer_segment.components.model_pusher import ModelPusher
import os, sys

from datetime import datetime
import pandas as pd


class Pipeline():

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise CustomException(e, sys) from e

    # data_ingestion

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys) from e
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact,
                                       data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config = self.config.get_data_transformation_config(),
                data_validation_artifact = data_validation_artifact)

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e,sys) from e

    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),

                                         data_transformation_artifact=data_transformation_artifact)

            return model_trainer.initiate_model_training()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_evaluation(self, data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact,
                               ):
        try:
            model_eval = ModelEvaluation(data_validation_artifact, model_trainer_artifact)

            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except  Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact):
        try:
            model_pusher = ModelPusher(model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except  Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            # data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact)

            model_trainer_artifact = self.start_model_training(
                data_transformation_artifact=data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)

        except Exception as e:
            raise CustomException(e, sys) from e
