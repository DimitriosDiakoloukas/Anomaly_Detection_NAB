# Anomaly_Detection_NAB
\documentclass{article}

\begin{document}

\title{Anomaly Detection with Machine Learning}
\author{Your Name}
\date{}

\maketitle

\section*{Overview}
Anomaly detection is a critical task in various industries, such as finance, manufacturing, and cybersecurity. This project aims to implement and compare different anomaly detection techniques using machine learning algorithms. The goal is to identify and flag unusual patterns or outliers in a given dataset, which can help in detecting potential issues, fraud, or anomalies in real-world scenarios.

\section*{Dependencies}
Before running the code, make sure you have the following dependencies installed:
\begin{itemize}
    \item pandas
    \item numpy
    \item matplotlib
    \item seaborn
    \item scikit-learn
    \item tensorflow
    \item keras
    \item pyemma
\end{itemize}

\section*{Dataset}
The project uses a real-world dataset containing time-series data of ambient temperature system readings. The dataset is provided in a ZIP archive, and it is read into a pandas DataFrame for further processing and analysis.

\section*{Techniques Implemented}
\begin{enumerate}
    \item \textbf{K-Means Clustering Anomaly Detection:} K-Means clustering is used to group data points into clusters, and the distance to cluster centroids is used as an anomaly score to detect outliers.
    
    \item \textbf{Cluster \& Gaussian Anomaly Detection:} Data points are clustered using K-Means, and a Gaussian distribution is fit to each cluster. Mahalanobis distance is calculated to identify anomalies based on the Gaussian distribution.
    
    \item \textbf{Markov Chain for Sequential Anomaly Detection:} The data is discretized, and a Markov Chain model is trained to capture sequential dependencies. Anomalies are detected using the transition probabilities.
    
    \item \textbf{Feedforward Neural Network (MLP) for Anomaly Detection:} A neural network with multiple hidden layers is implemented to classify data points as normal or anomalous based on features extracted from the dataset.
\end{enumerate}

\section*{How to Run}
\begin{enumerate}
    \item Download the ZIP archive containing the dataset and ensure it is placed at the correct file path.
    \item Install the required dependencies mentioned above using pip or any package manager of your choice.
    \item Run the main script to execute all the anomaly detection techniques and print the detected anomalies.
\end{enumerate}

\section*{License}
This project is open-source and is distributed under the \textit{MIT License}. Feel free to modify and use the code as per the terms of the license.

\section*{Note}
It's recommended to run the code in a virtual environment to avoid conflicts with existing packages in your system. Additionally, you can modify hyperparameters and experiment with different techniques to optimize the anomaly detection performance based on your specific use case.

\section*{Acknowledgments}
Special thanks to the authors and contributors of the libraries used in this project for their valuable work and support in the field of anomaly detection and machine learning.

\end{document}
