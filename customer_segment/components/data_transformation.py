import os
import sys
import pandas as pd
import numpy as np
from customer_segment.logger import logging
from customer_segment.exception import CustomException
from customer_segment.entity.artifact_entity import *
from customer_segment.entity.config_entity import *
from customer_segment.utils.utils import read_yaml_file, save_data, save_object
from customer_segment.constant import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class Feature_Engineering(BaseEstimator, TransformerMixin):

    def __init__(self, drop_columns):

        """
        This class applies necessary Feature Engneering
        """
        logging.info(f"\n{'*' * 20} Feature Engneering Started {'*' * 20}\n\n")

        ############### Accesssing Column Labels #########################

        #   Schema.yaml -----> Data Tranformation ----> Method: Feat Eng Pipeline ---> Class : Feature Eng Pipeline              #

        self.columns_to_drop = drop_columns

        ########################################################################

        logging.info(
            f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")

    # Feature Engineering Pipeline

    ######################### Data Modification ############################

    def drop_columns(self, X: pd.DataFrame):
        try:
            columns = X.columns

            logging.info(f"Columns before drop  {columns}")

            # Columns Dropping
            drop_column_labels = self.columns_to_drop

            logging.info(f" Dropping Columns {drop_column_labels} ")

            X = X.drop(columns=drop_column_labels, axis=1)

            return X

        except Exception as e:
            raise CustomException(e, sys) from e

    def drop_rows_with_nan(self, X: pd.DataFrame):
        # Log the shape before dropping NaN values
        logging.info(f"Shape before dropping NaN values: {X.shape}")

        # Drop rows with NaN values
        X = X.dropna()
        # X.to_csv("Nan_values_removed.csv", index=False)

        # Log the shape after dropping NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")

        logging.info("Dropped NaN values.")

        return X

    def drop_duplicates(self, X: pd.DataFrame):
        """
        Drops duplicate rows from a pandas DataFrame and returns the modified DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to remove duplicate rows from.

        Returns:
            pandas.DataFrame: The modified DataFrame with duplicate rows removed.
        """

        print(" Drop duplicate value")
        X = X.drop_duplicates()

        return X

    def separate_date_time(self, X, column_label):
        # Convert the column to datetime
        X[column_label] = pd.to_datetime(X[column_label])

        # Extract date and time from the given column
        X['Invoice_Date'] = X[column_label].dt.date
        X['Invoice_Date'] = pd.to_datetime(X['Invoice_Date'])

        # Drop the original column
        X.drop(column_label, axis=1, inplace=True)

        return X

    def remove_duplicate_rows_keep_last(self, X):

        logging.info(f"DataFrame shape before removing duplicates: {X.shape}")
        num_before = len(X)
        X.drop_duplicates(inplace=True)
        num_after = len(X)

        num_duplicates = num_before - num_after
        logging.info(f"Removed {num_duplicates} duplicate rows")
        logging.info(f"DataFrame shape after removing duplicates: {X.shape}")

        return X

    def convert_nan_null_to_nan(self, X: pd.DataFrame):
        # Convert "NAN" and "NULL" values to np.nan
        X.replace(["NAN", "NULL", "nan"], np.nan, inplace=True)

        # Return the updated DataFrame
        return X

    def calculate_total_price(self, X):
        try:
            X['TotalPrice'] = X['Quantity'] * X['UnitPrice']
        except KeyError:
            logging.error("One or more required columns (Quantity, UnitPrice) not found in the DataFrame.")
        except Exception as e:
            logging.error("An error occurred while calculating the Total Price: {}".format(str(e)))

        return X

    def drop_negative_rows(self, X, column_name):
        return X[X[column_name] >= 0]

    def get_max_min_dates(self, X: pd.DataFrame, date_column):
        max_date = X[date_column].max()
        min_date = X[date_column].min()
        return X, max_date, min_date

    def add_recency_column(self, X, date_column, min_date, max_date):

        X['recency'] = (max_date - X[date_column]).dt.days

        recency_table = X.groupby('CustomerID')['recency'].min().reset_index()
        logging.info(" Recency Table Created")
        logging.info(f"Shape of recency_table : {recency_table.shape}")

        return recency_table

    def add_frequency_column(self, X):
        frequency_table = X.groupby('CustomerID').count()['InvoiceNo'].to_frame().reset_index()
        frequency_table.rename(columns={'InvoiceNo': 'frequency'}, inplace=True)
        logging.info(" Frequency Table Created")
        logging.info(f"Shape of frequency_table : {frequency_table.shape}")
        return frequency_table

    def add_monetory_column(self, X):
        monetary_table = X.groupby('CustomerID')['TotalPrice'].sum().rename('monetary').reset_index()
        monetary_table.rename(columns={'TotalPrice': 'monetary'}, inplace=True)

        logging.info(" Monetory Table Created")
        logging.info(f"Shape of monetary_table : {monetary_table.shape}")
        return monetary_table

    def merge_tables(self, X, recency_table, frequency_table, monetary_table):
        logging.info("Merging to form RMF Table ...")

        data = X.groupby('CustomerID').first().reset_index()
        customer_ids = data['CustomerID']

        rfm_table = pd.merge(customer_ids, recency_table, on='CustomerID')
        rfm_table = pd.merge(rfm_table, frequency_table, on='CustomerID')
        rfm_table = pd.merge(rfm_table, monetary_table, on='CustomerID')

        logging.info(f"Tables merged : Columns - {rfm_table.columns}")
        return rfm_table

    def run_data_modification(self, data):

        X = data.copy()

        # Removing duplicated rows
        X = self.remove_duplicate_rows_keep_last(X)
        # Drop Columns
        X = self.drop_columns(X=data)

        # make Null as np.nan
        X = self.convert_nan_null_to_nan(X)

        # Drop rows with nan
        X = self.drop_rows_with_nan(X)

        # Drop negativve data from 'Quantity"
        logging.info(" Dropping rows with negative values")
        X = self.drop_negative_rows(X, 'Quantity')

        # Filtering data for negative data in Unit Price column
        logging.info(" Dropping rows with negative Unit Price ")
        X = X[X['UnitPrice'] >= 0]

        # Drop rows with nan
        X = self.drop_rows_with_nan(X)

        # Finding Total Price
        # Multiply 'Quantity' and 'UnitPrice' to calculate the total price

        logging.info("Calculating Total Price")
        X = self.calculate_total_price(X=X)

        # Separate Date and Time
        logging.info("Extracting Date and time")
        X = self.separate_date_time(X, 'InvoiceDate')
        logging.info(" Date and Time extracted from dataframe ")

        logging.info(f" Getting Earliest and recent Dates from the data ")
        X, max_date, min_date = self.get_max_min_dates(X, 'Invoice_Date')

        logging.info(f" Earliest Date : {min_date}")
        logging.info(f" Latest  Date : {max_date}")

        # Making Recency Column
        recency_table = self.add_recency_column(X, date_column='Invoice_Date',
                                                min_date=min_date, max_date=max_date)

        # recency_table.to_csv('Recency_check.csv')

        # Make Frequency Column

        frequency_table = self.add_frequency_column(X=X)

        # MAke Monetory Column

        monetory_table = self.add_monetory_column(X=X)

        # Rmf Table

        rmf_table = self.merge_tables(X=X, recency_table=recency_table,
                                      frequency_table=frequency_table,
                                      monetary_table=monetory_table)

        #  rmf_table.to_csv('rmf_table.csv')

        return rmf_table

        ######################### Outiers ############################

    def remove_outliers(self, data, columns, lower_threshold=0.05, upper_threshold=0.95):
        """
        Detects outliers using the quantile method and removes the outlier rows from the DataFrame.

        Args:
            data (pandas.DataFrame): The DataFrame containing the data.
            columns (list): A list of column names to check for outliers.
            lower_threshold (float, optional): The lower quantile threshold. Defaults to 0.05.
            upper_threshold (float, optional): The upper quantile threshold. Defaults to 0.95.

        Returns:
            pandas.DataFrame: The DataFrame with outlier rows removed.
        """
        # Copy the data to avoid modifying the original DataFrame
        data_cleaned = data.copy()

        # Iterate over the columns
        for col in columns:
            # Compute the lower and upper quantiles
            lower_quantile = data[col].quantile(lower_threshold)
            upper_quantile = data[col].quantile(upper_threshold)

            # Identify outlier rows
            outlier_rows = (data[col] < lower_quantile) | (data[col] > upper_quantile)

            # Count the number of outliers
            num_outliers = outlier_rows.sum()

            if num_outliers > 0:
                # Remove outlier rows
                data_cleaned = data_cleaned.loc[~outlier_rows]

                # Logging
                logging.info(f"Removed {num_outliers} outliers from column '{col}'.")
            else:
                logging.info(f"No outliers found in column '{col}'.")

        # Reset the index of the cleaned data
        data_cleaned.reset_index(drop=True, inplace=True)

        return data_cleaned

    def outlier(self, X):

        X = self.remove_outliers(X, columns=['recency', 'frequency', 'monetary'])

        return X

    def data_wrangling(self, X: pd.DataFrame):
        try:

            # Data Modification
            data_modified = self.run_data_modification(data=X)

            logging.info(" Data Modification Done")

            # Removing outliers

            logging.info(" Removing Outliers")

            df_outlier_removed = self.outlier(X=data_modified)

            return df_outlier_removed


        except Exception as e:
            raise CustomException(e, sys) from e

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        try:
            data_modified = self.data_wrangling(X)

            #   data_modified.to_csv("data_modified.csv",index=False)
            logging.info(" Data Wrangaling Done ")

            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shapde Modified Data : {data_modified.shape}")

            return data_modified
        except Exception as e:
            raise CustomException(e, sys) from e


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*' * 20} Data Transformation log started {'*' * 20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            ############### Accesssing Column Labels #########################

            #           Schema.yaml -----> DataTransfomation

            # Transformation Yaml File path

            # Reading data in Schema
            self.transformation_yaml = read_yaml_file(file_path=TRANSFORMATION_YAML_FILE_PATH)

            self.drop_columns = self.transformation_yaml[DROP_COLUMNS]

        # self.drop_columns=self.schema[DROP_COLUMN_KEY]

        ########################################################################
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_feature_engineering_object(self):
        try:

            feature_engineering = Pipeline(steps=[("fe", Feature_Engineering(drop_columns=self.drop_columns))])
            return feature_engineering
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_transformer_object(self):
        try:
            logging.info('Creating Data Transformer Object')
            numerical_col = ['recency', 'frequency', 'monetary']

            numerical_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()),
            ])
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_col)
            ])
            return preprocessor


        except Exception as e:
            logging.error('An error occurred during data transformation')
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self):
        try:
            # Data validation Artifact ------>Accessing train and test files
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_validation_artifact.validated_train_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)

            logging.info(f" Accesig train and test data \
                         Train Data : {train_file_path}")

            logging.info(f" Traning columns {train_df.columns}")

            # Feature Engineering
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()

            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            train_df = fe_obj.fit_transform(train_df)

            # logging.info(f" Columns in feature enginering {feature_eng_test_df.columns}")
            logging.info(f"Saving feature engineered training  dataframe.")
            transformed_train_dir = self.data_transformation_config.transformed_train_dir

            Feature_eng_train_file_path = os.path.join(transformed_train_dir, "Feature_engineering.csv")

            save_data(file_path=Feature_eng_train_file_path, data=train_df)

            ############ Input Fatures transformation########
            ## Preprocessing
            logging.info("*" * 20 + " Applying preprocessing object on training dataframe  " + "*" * 20)
            preprocessing_obj = self.get_data_transformer_object()
            train_arr = preprocessing_obj.fit_transform(train_df)
            # Log the shape of train_arr
            logging.info(f"Shape of train_arr: {train_arr.shape}")

            logging.info("Transformation completed successfully")

            col = ['recency', 'frequency', 'monetary']

            transformed_train_df = pd.DataFrame(train_arr, columns=col)

            # Customer ID
            transformed_train_df['CustomerID'] = train_df['CustomerID']

            # Saving transformed data
            transformed_train_dir = self.data_transformation_config.transformed_train_dir

            transformed_train_file_path = os.path.join(transformed_train_dir, "transformed_train.csv")

            ###############################################################

            # Saving the Transformed array ----> csv
            ## Saving transformed train  file
            logging.info("Saving Transformed Train file")

            save_data(file_path=transformed_train_file_path, data=transformed_train_df)

            logging.info("Transformed Train file saved")
            logging.info("Saving Feature Engineering Object")

            ### Saving FFeature engineering and preprocessor object
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path=feature_engineering_object_file_path, obj=fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR, PIKLE_FOLDER_NAME_KEY,
                                               os.path.basename(feature_engineering_object_file_path)), obj=fe_obj)

            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path=preprocessing_object_file_path, obj=preprocessing_obj)
            save_object(file_path=os.path.join(ROOT_DIR, PIKLE_FOLDER_NAME_KEY,
                                               os.path.basename(preprocessing_object_file_path)), obj=preprocessing_obj)

            # Feature_eng_train_file_path
            Feature_eng_train_file_path = Feature_eng_train_file_path

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformation successfull.",
                                                                      Feature_eng_train_file_path=Feature_eng_train_file_path,
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      preprocessed_object_file_path=preprocessing_object_file_path,
                                                                      feature_engineering_object_file_path=feature_engineering_object_file_path)

            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def __del__(self):
        logging.info(f"\n{'*' * 20} Data Transformation log completed {'*' * 20}\n\n")