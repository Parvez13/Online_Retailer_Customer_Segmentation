import ast
import logging
import sys
import time
import os
import pandas as pd
from customer_segment.logger import logging
from customer_segment.exception import CustomException
from customer_segment.utils.utils import save_object, read_yaml_file
from customer_segment.entity.config_entity import ModelTrainerConfig
from customer_segment.entity.artifact_entity import DataTransformationArtifact
from customer_segment.entity.artifact_entity import ModelTrainerArtifact
from customer_segment.constant import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import silhouette_score
import yaml
import shutil


class Model:
    def __init__(self, model_png_location):

        self.model_png_location = model_png_location
        file_location = self.model_png_location
        os.makedirs(file_location, exist_ok=True)

    # Kmeans
    def choose_clusters(self, df, max_clusters=50):
        silhouette_scores = []

        try:
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
                kmeans.fit(df)
                labels = kmeans.predict(df)
                silhouette_avg = silhouette_score(df, labels)
                silhouette_scores.append(silhouette_avg)
        except Exception as e:
            raise CustomException(e, sys) from e

        optimal_clusters = np.argmax(silhouette_scores) + 2

        return optimal_clusters

    def perform_kmeans_clustering(self, df, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', algorithm='auto')
        kmeans.fit(df)
        labels = kmeans.predict(df)
        centroids = kmeans.cluster_centers_
        rfm_kmean = df.copy()

        rfm_kmean['cluster'] = labels

        logging.info(" Kmeans Fitted ")
        return rfm_kmean, kmeans

    def save_rfm_plot(self, rfm_df, directory, model_name, cluster_column, fea_eng_data):
        # Extract RFM components and cluster labels
        rfm_df[['recency', 'frequency', 'monetary']] = fea_eng_data[['recency', 'frequency', 'monetary']]
        cluster_labels = rfm_df[cluster_column]

        # Set a color palette for cluster visualization
        num_clusters = len(cluster_labels.unique())
        colors = sns.color_palette('viridis', num_clusters)

        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

        # Plot frequency vs monetary relationship for each cluster
        for cluster_label, color in zip(cluster_labels.unique(), colors):
            cluster_data = rfm_df[cluster_labels == cluster_label]
            axes.scatter(
                cluster_data['frequency'],
                cluster_data['monetary'],
                color=color,
                label=f'{cluster_label}'
            )

        # Set labels and title for the plot
        axes.set_xlabel('Frequency')
        axes.set_ylabel('Monetary')
        axes.set_title('Frequency vs Monetary')
        axes.legend()

        plt.tight_layout()

        # Create directory if it doesn't exist
        file_path = os.path.join(directory, model_name)
        os.makedirs(file_path, exist_ok=True)

        # Save the figure
        filename = os.path.join(file_path, 'prediction.png')
        fig.savefig(filename)

        return filename

    def Kmeans_train(self, data):
        # Choosing Optimal cluster

        #  optimal_cluster=self.choose_clusters(df=data)

        optimal_cluster = 4  # Based on EDA
        logging.info(f" Based on silhouette_score optimal number of clusters ---K_MEANS--- : {optimal_cluster}")

        rfm_kmean, kmeans_model = self.perform_kmeans_clustering(df=data, n_clusters=optimal_cluster)

        # rfm_kmean.to_csv('rfm_kmean.csv')

        return rfm_kmean, kmeans_model, optimal_cluster

    # Gaussian mixture clutsering
    def find_optimal_clusters(self, data, max_clusters):
        """
        Find the optimal number of clusters using Gaussian Mixture Model clustering and silhouette scores.

        Args:
            data (array-like): The input data for clustering.
            max_clusters (int): The maximum number of clusters to consider.

        Returns:
            int: The optimal number of clusters.
        """
        if max_clusters < 3:
            raise ValueError("max_clusters must be at least 3")

        silhouette_scores = []  # Initialize an empty list

        for k in range(3, max_clusters + 1):  # Start from 3 clusters
            # Perform Gaussian Mixture Model clustering
            model = GaussianMixture(n_components=k,
                                    init_params='k-means++')
            model.fit(data)
            labels = model.predict(data)

            # Compute the silhouette score
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)

        # Find the optimal number of clusters based on the highest silhouette score
        optimal_clusters, _ = max(enumerate(silhouette_scores), key=lambda x: x[1])

        logging.info(f"Optimal Clusters in GMM: {optimal_clusters + 3}")

        return optimal_clusters + 3

    def GaussianMixtureClustering(self, data, optimal_clusters):
        # Perform Gaussian Mixture Model clustering with the desired number of clusters
        model = GaussianMixture(n_components=optimal_clusters, init_params='k-means++', covariance_type='spherical')
        model.fit(data)
        labels = model.predict(data)

        logging.info("Labels created")

        # Return the cluster labels and the trained model
        return labels, model

    def adding_labels_to_data(self, data, cluster_labels):
        # Convert cluster_labels to a DataFrame
        logging.info("Converting cluster_labels to DataFrame")
        df_cluster_labels = pd.DataFrame(cluster_labels, columns=['cluster'])

        # Add cluster labels to the data
        data['cluster'] = df_cluster_labels['cluster']
        logging.info("Cluster added to the DataFrame")

        return data

    def GaussianMixtureClusteringTrain(self, data):
        # Optimal number of clusters
        optimal_cluster = self.find_optimal_clusters(data=data, max_clusters=4)

        # Model Training
        logging.info("Performing Gaussian Mixture Model clustering")
        labels, gmm_model = self.GaussianMixtureClustering(data=data, optimal_clusters=optimal_cluster)

        # Adding cluster labels to the data
        logging.info("Adding cluster labels to the data")
        data = self.adding_labels_to_data(data=data, cluster_labels=labels)

        rfm_gmm_table = data
        # rfm_gmm_table.to_csv('rfm_gmm_table.csv')

        return rfm_gmm_table, gmm_model, optimal_cluster


class ModelTrainer:

    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")

            # Accessing Artifacts
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            ## Schema Yaml
            self.schema_data = read_yaml_file(SCHEMA_FILE_PATH)

            # Model.yaml path
            self.model_config_path = self.model_trainer_config.model_config_path




        except Exception as e:
            raise CustomException(e, sys) from e

    def compare_cluster_labels(self, kmeans_table, gmm_table, data, optimal_cluster_kmean, optimal_cluster_gmm,
                               k_means_model, kmean_prediction_png,
                               gmm_model, gmm_prediction_png):
        """
        Compare two sets of cluster labels (K-means and Gaussian Mixture Model) using evaluation metrics and return the selected model name.

        Args:
            kmeans_labels (array-like): Cluster labels assigned by K-means.
            gmm_labels (array-like): Cluster labels assigned by Gaussian Mixture Model.
            data (array-like): The input data for clustering.

        Returns:
            str: The name of the selected model ('K-means' or 'GMM').
            str: The name of the model selected based on the evaluation metrics.
        """
        kmeans_table = kmeans_table
        k_means_model = k_means_model
        kmean_prediction_png = kmean_prediction_png
        optimal_cluster_kmean = optimal_cluster_kmean
        kmeans_labels = kmeans_table['cluster']

        gmm_table = gmm_table
        gmm_model = gmm_model
        gmm_prediction_png = gmm_prediction_png
        optimal_cluster_gmm = optimal_cluster_gmm
        gmm_labels = gmm_table['cluster']

        # Compute evaluation metrics
        silhouette_kmeans = silhouette_score(data, kmeans_labels)
        silhouette_gmm = silhouette_score(data, gmm_labels)

        # Compare evaluation metrics
        if (silhouette_kmeans >= silhouette_gmm):
            return 'K-means', k_means_model, silhouette_kmeans, kmean_prediction_png, optimal_cluster_kmean, kmeans_table
        else:
            return 'GMM', gmm_model, silhouette_gmm, gmm_prediction_png, optimal_cluster_gmm, gmm_table

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Finding transformed Training data")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path

            logging.info("Transformed Data found!!! Now, converting it into dataframe")
            train_df = pd.read_csv(transformed_train_file_path)
            # making rfm Table
            train_rfm_table = train_df[['recency', 'frequency', 'monetary']]

            # Train data --> Feature engineererd
            train_data = self.data_transformation_artifact.Feature_eng_train_file_path
            train_data_df = pd.read_csv(train_data)
            logging.info(" Training Kmeans.....")
            model = Model(model_png_location=self.model_trainer_config.png_location)

            # Kmeans
            rfm_kmean_table, kmeans_model, optimal_cluster_kmean = model.Kmeans_train(data=train_rfm_table)

            prediction_png_kmeans = model.save_rfm_plot(rfm_df=rfm_kmean_table, model_name='k_mean_prediction_data',
                                                        directory=self.model_trainer_config.png_location,
                                                        cluster_column='cluster', fea_eng_data=train_data_df)

            logging.info(" Training GMM Clustering.....")

            # GMM
            rfm_gmm_table, gmm_model, optimal_cluster_gmm = model.GaussianMixtureClusteringTrain(data=train_rfm_table)

            prediction_png_gmm = model.save_rfm_plot(rfm_df=rfm_gmm_table,
                                                     directory=self.model_trainer_config.png_location,
                                                     model_name='gmm_cluster_prediction_data', cluster_column='cluster',
                                                     fea_eng_data=train_data_df)

            # Evaluating
            logging.info(" Evaluating .....")
            model_name, model, silhouette_score, prediction_png_path, optimal_cluster, rfm_table = self.compare_cluster_labels(
                kmeans_table=rfm_kmean_table,
                gmm_table=rfm_gmm_table,
                k_means_model=kmeans_model,
                kmean_prediction_png=prediction_png_kmeans,
                optimal_cluster_kmean=optimal_cluster_kmean,
                gmm_prediction_png=prediction_png_gmm,
                optimal_cluster_gmm=optimal_cluster_gmm,
                gmm_model=gmm_model,
                data=train_rfm_table)

            # Model Name
            logging.info(f" Model Selected :{model_name}")
            logging.info(f"-------------")

            # Saving rfm tables of the selcted model
            rfm_table['CustomerID'] = train_df['CustomerID']
            rfm_table[['recency', 'frequency', 'monetary']] = train_data_df[['recency', 'frequency', 'monetary']]

            trained_model_directory = self.model_trainer_config.trained_model_directory
            os.makedirs(trained_model_directory, exist_ok=True)
            logging.info(f"Saving rfm csv at path: {trained_model_directory}")
            csv_file_path = os.path.join(trained_model_directory, 'rfm.csv')
            logging.info(f" rfm table Columns : {rfm_table.columns}")
            rfm_table.to_csv(csv_file_path)

            # Saving Model.pkl object
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=model)

            # Saving Report
            best_model_name = model_name
            logging.info(f"Saving metrics of model  : {best_model_name}")
            report = {
                "Model_name": best_model_name,
                "Silhouette_score": str(silhouette_score),
                "number_of_clusters": str(optimal_cluster)
            }

            logging.info(f"Dumping Metrics in report.....")

            # Save report in artifact folder
            model_artifact_report_path = self.model_trainer_config.report_path
            report_file_path = os.path.join(model_artifact_report_path, 'report.yaml')

            os.makedirs(model_artifact_report_path, exist_ok=True)

            with open(report_file_path, 'w') as file:
                yaml.safe_dump(report, file)

            logging.info("Report Created")

            # Saving selected prediction image
            trained_model_directory = self.model_trainer_config.trained_model_directory
            shutil.copy(prediction_png_path, trained_model_directory)
            logging.info(" Copied prediction png file to the desired directory")

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model Trained successfully",
                model_selected=trained_model_file_path,
                model_name=model_name,
                report_path=report_file_path,
                model_prediction_png=prediction_png_path,
                csv_file_path=csv_file_path

            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")

