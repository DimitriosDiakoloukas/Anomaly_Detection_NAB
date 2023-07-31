# Anomaly Detection with Machine Learning

Anomaly detection is a critical task in various industries, such as finance, manufacturing, and cybersecurity. This project aims to implement and compare different anomaly detection techniques using machine learning algorithms. The goal is to identify and flag unusual patterns or outliers in a given dataset, which can help in detecting potential issues, fraud, or anomalies in real-world scenarios.

## Dependencies
Before running the code, make sure you have the following dependencies installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
- pyemma

## Dataset
The project uses a real-world dataset containing time-series data of ambient temperature system readings. The dataset is provided in a ZIP archive, and it is read into a pandas DataFrame for further processing and analysis. The dataset is downloaded from Kaggle (Numenta Anomaly Benchmark (NAB)).

## Techniques Implemented
1. **K-Means Clustering Anomaly Detection:** K-Means clustering is used to group data points into clusters, and the distance to cluster centroids is used as an anomaly score to detect outliers.

2. **Cluster & Gaussian Anomaly Detection:** Data points are clustered using K-Means, and a Gaussian distribution is fit to each cluster. Mahalanobis distance is calculated to identify anomalies based on the Gaussian distribution.

3. **Markov Chain for Sequential Anomaly Detection:** The data is discretized, and a Markov Chain model is trained to capture sequential dependencies. Anomalies are detected using the transition probabilities.

4. **Feedforward Neural Network (MLP) for Anomaly Detection:** A neural network with multiple hidden layers is implemented to classify data points as normal or anomalous based on features extracted from the dataset.

## How to Run
1. Download the ZIP archive containing the dataset and ensure it is placed at the correct file path.
2. Install the required dependencies mentioned above using pip or any package manager of your choice.
3. Run the main script to execute all the anomaly detection techniques and print the detected anomalies.

## License
This project is open-source and is distributed under the MIT License. Feel free to modify and use the code as per the terms of the license.

## Note
It's recommended to run the code in a virtual environment to avoid conflicts with existing packages in your system. Additionally, you can modify hyperparameters and experiment with different techniques to optimize the anomaly detection performance based on your specific use case.

## Acknowledgments
Special thanks to the authors and contributors of the libraries used in this project for their valuable work and support in the field of anomaly detection and machine learning.
