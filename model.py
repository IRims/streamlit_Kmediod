import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(csv_name):
    return pd.read_csv(csv_name)

def perform_kmedoids_clustering(df, n_clusters):
    # Separate numeric and non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Scale numeric data
    scaler = StandardScaler()
    Clus_dataSet = scaler.fit_transform(numeric_df)
    
    # Initialize medoids
    initial_medoids = np.random.choice(range(len(Clus_dataSet)), n_clusters, replace=False)
    
    # Create instance of K-Medoids algorithm using Euclidean distance metric
    kmedoids_instance = kmedoids(Clus_dataSet, initial_medoids, metric=distance_metric(type_metric.EUCLIDEAN))
    
    # Run cluster analysis and obtain results
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    
    # Assign cluster labels
    labels = np.zeros(len(df))
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id
            
    df['kmedoids Cluster Labels'] = labels
    X_scaled = scaler.transform(numeric_df)
    return df, X_scaled, "Clustering done successfully"

