from customer_segment.logger import logging
from customer_segment.pipeline.pipeline import Pipeline
def main():
    try:
        # Instance Pipeline
        instance_pipeline = Pipeline()
        instance_pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__ == "__main__":
    main()