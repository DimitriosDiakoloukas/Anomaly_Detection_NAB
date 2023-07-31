import pandas as pd
import numpy as np
import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam
import time
import sys
import pyemma
from pyemma import msm 
import zipfile
import os

# Provide the correct file path to the ZIP archive on Windows
zip_file_path_windows = r"data/archive.zip"

# Provide the relative path to the CSV file within the ZIP archive
csv_file_inside_zip = "realKnownCause/realKnownCause/ambient_temperature_system_failure.csv"

# Verify if the ZIP archive exists at the provided file path
if not os.path.exists(zip_file_path_windows):
    raise FileNotFoundError("The ZIP archive does not exist at the specified location.")

# Create a directory to extract the ZIP contents in WSL
extraction_dir = os.path.join(os.getcwd(), "data/extracted_files")
os.makedirs(extraction_dir, exist_ok=True)

# Extract the CSV file from the ZIP archive on Windows
with zipfile.ZipFile(zip_file_path_windows, 'r') as zip_ref:
    zip_ref.extract(csv_file_inside_zip, path=extraction_dir)

# Provide the correct file path for the extracted CSV file in WSL
csv_file_path_wsl = os.path.join(extraction_dir, csv_file_inside_zip)

# Read the CSV file after extraction in WSL
df = pd.read_csv(csv_file_path_wsl)

# print(df.info()) 

# print("pyemma version:", pyemma.__version__)

# print(df['timestamp'].head(15))
# print(df['value'].mean())

# Convert the 'timestamp' column to pandas datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Create the categories based on weekdays/weekends and day/night
def create_category(row):
    if row['day_of_week'] < 5:  # Monday (0) to Friday (4) are weekdays
        if 6 <= row['hour'] < 18:  # 6 AM to 5:59 PM are daytime hours
            return 'weekday_day'
        else:  # 6 PM to 5:59 AM are nighttime hours
            return 'weekday_night'
    else:  # Saturday (5) and Sunday (6) are weekends
        if 6 <= row['hour'] < 18:  # 6 AM to 5:59 PM are daytime hours
            return 'weekend_day'
        else:  # 6 PM to 5:59 AM are nighttime hours
            return 'weekend_night'

# Apply the function to create the 'category' column
df['category'] = df.apply(create_category, axis=1)

# Include the 'hour' and 'day_of_week' columns in the 'data' DataFrame
data = df[['value', 'hour', 'day_of_week', 'category']]

# Drop the intermediate 'hour' and 'day_of_week' columns if you don't need them anymore
df.drop(['hour', 'day_of_week'], axis=1, inplace=True)

# One-hot encode the 'category' column
data = pd.get_dummies(data, columns=['category'])

# Remove Comments to Plot the histogram
'''
plt.figure(figsize=(8, 6))
plt.hist(df['category'], bins=4, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Categories', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
'''

############################################
### K-MEANS CLUSTERING ANOMALY DETECTION ###
############################################
def kmeans_anomaly_detection(data, anomaly_threshold=2.5):
    min_max_scaler = preprocessing.StandardScaler()
    scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(scaled)

    # Reduce to 2 important features using PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Elbow Method to find optimal number of clusters
    distortions = []

    # Maximum number of clusters to consider
    max_clusters = 10  

    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)

        # Inertia is the sum of squared distances to the nearest centroid
        distortions.append(kmeans.inertia_) 

    # Remove Comments to Plot the elbow curve
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()
    '''

    # Determine the optimal number of clusters based on the elbow point
    # Chose the number of clusters based on the elbow point from the plot
    optimal_num_clusters = 2 

    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(data)

    # Calculate distance to cluster centroid
    df['distance_to_centroid'] = kmeans.transform(data).min(axis=1)

    # Already defined anomaly threshold (you can set this based the data distribution)
    # Identify anomalies
    anomalies = df[df['distance_to_centroid'] > anomaly_threshold]

    # Function to perform K-means clustering with 15 centroids
    def kmeans_clustering(data, num_clusters=15):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(data)
        data['distance_to_centroid'] = kmeans.transform(data.drop('cluster', axis=1)).min(axis=1)
        return data

    # Perform K-means clustering with 15 centroids and print the result
    data_with_clusters = kmeans_clustering(pd.DataFrame(data_pca), num_clusters=15)
    #print(data_with_clusters)

    data_with_clusters['PCA1'] = data_pca[:, 0]
    data_with_clusters['PCA2'] = data_pca[:, 1]
        
    # Print the anomalies
    print("Anomalies using K-means clustering:")
    return anomalies

# Remove comment to print anomaly
#print(kmeans_anomaly_detection(data, anomaly_threshold=2.5))

############################################
### CLUSTER & GAUSSIAN ANOMALY DETECTION ###
############################################

def cluster_gaussian_anomaly_detection(data, num_clusters=2, mahalanobis_threshold=3.0):
    # Function to fit Gaussian distribution to each cluster's data
    def fit_cluster_gaussian(cluster_data):
        cluster_mean = cluster_data.mean(axis=0)
        cluster_cov = cluster_data.cov()
        return cluster_mean, cluster_cov

    # Function to calculate Mahalanobis distance from Gaussian distribution
    def calculate_mahalanobis_distance(data_point, cluster_mean, cluster_cov):
        diff = data_point - cluster_mean
        inv_cov = np.linalg.inv(cluster_cov)
        mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
        return mahalanobis_distance

    # Perform K-means clustering to get cluster assignments
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data)

    # Reduce data to 2 important features using PCA
    data_with_clusters = pd.DataFrame(data=PCA(n_components=2).fit_transform(data), columns=['PCA1', 'PCA2'])

    # Dictionary to store Gaussian parameters for each cluster
    gaussian_parameters = {}
    for cluster_id in range(num_clusters):
        cluster_data = data_with_clusters[data['cluster'] == cluster_id]
        if not cluster_data.empty:
            cluster_mean, cluster_cov = fit_cluster_gaussian(cluster_data)
            gaussian_parameters[cluster_id] = (cluster_mean, cluster_cov)

    # Calculate Mahalanobis distance for each data point and identify anomalies
    data_with_clusters['mahalanobis_distance'] = 0
    for index, row in data_with_clusters.iterrows():
        data_point = row[['PCA1', 'PCA2']]
        cluster_id = data['cluster'].iloc[index]
        if cluster_id in gaussian_parameters:
            cluster_mean, cluster_cov = gaussian_parameters[cluster_id]
            mahalanobis_distance = calculate_mahalanobis_distance(data_point, cluster_mean, cluster_cov)
            data_with_clusters.at[index, 'mahalanobis_distance'] = mahalanobis_distance

    # Identify anomalies using Mahalanobis distance and threshold
    anomalies_cluster_gaussian = data_with_clusters[data_with_clusters['mahalanobis_distance'] > mahalanobis_threshold]
    # Print the anomalies
    #print("Anomalies using Cluster & Gaussian:")
    return anomalies_cluster_gaussian

# Remove comment to print anomaly
#print(cluster_gaussian_anomaly_detection(data, num_clusters=2, mahalanobis_threshold=3.0))

#####################################################
### MARKOV CHAIN FOR SEQUENTIAL ANOMALY DETECTION ###
#####################################################

def markov_anomaly_detection(df, window_size=5, threshold=0.99):
    # Step 1: Discretize Data Points
    #state_map = {'VL': 0, 'L': 1, 'A': 2, 'H': 3, 'VH': 4}

    def discretize_value(value):
        if value < 65:
            return 0  # 'VL'
        elif value < 70:
            return 1  # 'L'
        elif value < 75:
            return 2  # 'A'
        elif value < 80:
            return 3  # 'H'
        else:
            return 4  # 'VH'

    # Step 2: Define States for Markov Chain (based on 'value' column)
    states = [0, 1, 2, 3, 4]

    # Step 3: Train Markov Model and Get Transition Matrix
    def get_matrix_transition(df):
        df['state'] = df['value'].apply(discretize_value)
        df['next_state'] = df['state'].shift(-1)

        # Remove rows with missing values in 'next_state'
        df = df.dropna(subset=['next_state'])

        unique_states = np.unique(df[['state', 'next_state']].values)
        num_states = len(unique_states)

        # Create a transition matrix with all zeros
        transition_matrix = np.zeros((num_states, num_states))

        # Count transitions between states
        for i in range(1, len(df)):
            current_state = df['state'].iloc[i]
            next_state = df['next_state'].iloc[i]
            current_state_index = np.where(unique_states == current_state)[0][0]
            next_state_index = np.where(unique_states == next_state)[0][0]
            transition_matrix[current_state_index, next_state_index] += 1

        # Normalize transition probabilities
        transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)

        return transition_matrix

    # Step 4: Detect Anomalies with Markov Chain
    def markov_anomaly(df, window_size1, threshold1):
        transition_matrix = get_matrix_transition(df)
        real_threshold = threshold1 ** window_size1

        # Initialize the anomaly list with zeros for the entire DataFrame
        df['anomaly'] = [0] * len(df)

        for j in range(window_size1, len(df)):
            # Discretize the values and create the sequence
            sequence = df['value'][j - window_size1:j].apply(discretize_value).tolist()

            # Calculate the transition probabilities of the sequence based on the transition matrix
            probabilities = [transition_matrix[states.index(sequence[i])][states.index(sequence[i + 1])] for i in range(window_size1 - 1)]

            # Determine if any of the probabilities in the sequence is below the threshold
            is_anomaly = any(prob < real_threshold for prob in probabilities)

            # Store the anomaly result in the 'anomaly' column
            df['anomaly'].iloc[j] = 1 if is_anomaly else 0

        return df['anomaly']

    # Window size and threshold being set as parameters
    # Detect anomalies using Markov Chain method
    anomalies_markov = markov_anomaly(df, window_size, threshold)

    # Print the detected anomalies using Markov Chain method
    print("Anomalies (Markov Chain):")
    return anomalies_markov

# Remove comment to print anomaly
#print(markov_anomaly_detection(df, window_size=5, threshold=0.99))

##############################################################
### FEEDFORWARD NEURAL NETWORK (MLP) FOR ANOMALY DETECTION ###
##############################################################

def neural_network_anomaly_detection(X_train, anomaly_indices_df, threshold=0.5):
    # Convert X_train to a NumPy array
    X_train_array = X_train.select_dtypes(include=[np.number]).values

    # Prepare the labels for neural network training
    y_train = np.zeros(len(X_train_array))  # Set all labels to 0 (non-anomalies)

    # Extract the index values from the DataFrame of anomaly_indices
    anomaly_indices = anomaly_indices_df.index

    # Set the labels of anomalies to 1
    y_train[anomaly_indices] = 1

    # Define the neural network architecture
    model = Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=X_train_array.shape[1]))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_array, y_train, batch_size=32, epochs=10, validation_split=0.2)

    # Remove Comments to Plot the training and validation loss
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    '''

    # Set the anomaly threshold based on validation data in parameters
    # You can adjust this value based on validation results in parameters
    # Detect anomalies using the neural network
    y_pred_prob = model.predict(X_train_array)
    anomalies_neural_network = X_train[y_pred_prob > threshold] 

    # Output the anomalies
    print("Anomalies using Neural Network:")
    return anomalies_neural_network

# Call the function with the correct arguments
anomaly_indices_df = kmeans_anomaly_detection(data, anomaly_threshold=2.5)
print(neural_network_anomaly_detection(X_train=data, anomaly_indices_df=anomaly_indices_df, threshold=0.5))