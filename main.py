from customer_segment.exception import CustomException
from customer_segment.logger import logging
from customer_segment.config.configuration import Configuration
from customer_segment.components.data_ingestion import DataIngestion
from customer_segment.pipeline.pipeline import Pipeline
import os

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__ == "__main__":
    main()