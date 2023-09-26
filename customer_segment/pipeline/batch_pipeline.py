import os
import logging
from customer_segment.logger import logging
from customer_segment.exception import CustomException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from customer_segment.utils.utils import read_yaml_file, load_object
from customer_segment.constant import *
from customer_segment.constant import *
import os

PREDICTION_FOLDER = 'batch_Prediction'
PREDICTION_CSV = 'prediction_csv'
PREDICTION_FILE = 'prediction.csv'

FEATURE_ENG_FOLDER = 'feature_eng'

ROOT_DIR = os.getcwd()
FEATURE_ENG = os.path.join(ROOT_DIR, PREDICTION_FOLDER, FEATURE_ENG_FOLDER)
BATCH_PREDICTION = os.path.join(ROOT_DIR, PREDICTION_FOLDER, PREDICTION_CSV)
BATCH_COLLECTION_PATH = 'batch_prediction'


class batch_prediction:
    def __init__(self, input_file_path,
                 model_file_path,
                 transformer_file_path,
                 feature_engineering_file_path) -> None:
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

    def start_batch_prediction(self):
        logging.info("Loading the saved pipeline")

        # Load the feature engineering pipeline
        with open(self.feature_engineering_file_path, 'rb') as f:
            feature_pipeline = pickle.load(f)

        logging.info(f"Feature eng Object acessed :{self.feature_engineering_file_path}")

        # Load the data transformation pipeline
        with open(self.transformer_file_path, 'rb') as f:
            preprocessor = pickle.load(f)

        logging.info(f"Preprocessor  Object acessed :{self.transformer_file_path}")

        # Load the model separately
        model = load_object(file_path=self.model_file_path)

        logging.info(f"Model File Path: {self.model_file_path}")

        # Create the feature engineering pipeline
        feature_engineering_pipeline = Pipeline([
            ('feature_engineering', feature_pipeline)
        ])

        # Read the input file
        df = pd.read_csv(self.input_file_path)

        # Customer_ids
        ID = pd.DataFrame()
        ID['CustomerID'] = df['CustomerID']

        # Apply feature engineering
        df = feature_engineering_pipeline.transform(df)

        # Save the feature-engineered data as a CSV file
        FEATURE_ENG_PATH = FEATURE_ENG  # Specify the desired path for saving the file
        os.makedirs(FEATURE_ENG_PATH, exist_ok=True)
        file_path = os.path.join(FEATURE_ENG_PATH, 'batch_fea_eng.csv')
        df.to_csv(file_path, index=False)
        logging.info("Feature-engineered batch data saved as CSV.")

        # df.to_csv('dropped_strength.csv')

        logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")

        # Transform the feature-engineered data using the preprocessor
        transformed_data = preprocessor.transform(df)
        logging.info(f"Transformed Data Shape: {transformed_data.shape}")

        logging.info("Transformation completed successfully")

        col = ['recency', 'frequency', 'monetary']
        transformed_train_df = pd.DataFrame(transformed_data, columns=col)

        # Saving preprocessed dataframe
        transformed_train_df.to_csv(file_path, index=False)

        predictions = model.predict(transformed_train_df)
        logging.info(f"Predictions done :{predictions}")

        # Create a DataFrame from the predictions array
        df_predictions = pd.DataFrame(predictions, columns=['cluster'])

        # Adding cluster labels to the Dataframe
        transformed_train_df['cluster'] = df_predictions['cluster']

        # Addding corresponding Customer ids
        transformed_train_df['CustomerID'] = ID['CustomerID']

        # Save the predictions to a CSV file
        BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
        os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
        csv_path = os.path.join(BATCH_PREDICTION_PATH, 'predictions.csv')

        transformed_train_df.to_csv(csv_path)
        logging.info(f"Batch predictions saved to '{csv_path}'.")

        return csv_path



