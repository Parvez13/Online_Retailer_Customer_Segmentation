from collections import namedtuple

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["dataset_download_url",
                                  "ingested_data_dir",
                                  "raw_data_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                  ["schema_file_path", "validated_train_path"])
