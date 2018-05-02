# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:41:25 2018

@author: tharun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('OnlineRetail.xlsx')
data = dataset.iloc[:, 6:8].values
data[:, 0] = pd.to_datetime(data[:, 0])
'''
data[:, 0].strftime('%m/%d/%Y')
type(data['InvoiceDate'])
'''
'''
X1 = pd.to_datetime(dataset.iloc[:, 6].values, format="%d/%m/%Y")
X2 = dataset.iloc[:, 7].values.reshape(541909,1)
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data = sc_X.fit_transform(data)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters =4 , init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(data)


plt.scatter(data[y_kmeans == 0, 0], data[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data[y_kmeans == 1, 0], data[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(data[y_kmeans == 2, 0], data[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(data[y_kmeans == 3, 0], data[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
axes = plt.gca()
axes.set_xlim([-2.0,1.5])
axes.set_ylim([-4.0,3.5])
plt.title('Clusters of customers')
plt.xlabel('Timestamp')
plt.ylabel('Spent Amount')
plt.legend()
plt.show()



