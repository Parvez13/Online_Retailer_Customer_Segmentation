from customer_segment.logger import logging
from customer_segment.pipeline.instance_pipeline import Pipeline
from customer_segment.pipeline.batch_pipeline import batch_prediction
def main():
    try:
        # Instance Pipeline
        # instance_pipeline = Pipeline()
        # instance_pipeline.run_pipeline()
        # Batch Pipeline
        batch_pipeline = batch_prediction()
        batch_pipeline.start_batch_prediction()
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__ == "__main__":
    main()