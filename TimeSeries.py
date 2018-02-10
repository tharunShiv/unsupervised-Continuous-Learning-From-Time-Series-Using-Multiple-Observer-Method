# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_excel('dataset-DR.xlsx')
#dropping columns
data = data.drop(data.columns[[1,2,3,7]], axis=1)
#Now data is of DataFrame format, make it into sliceable format
X = data.iloc[:, 1:].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, 38:41] = sc.fit_transform(X[:, 38:41])



# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)   # first run with the n_components=None , then after seeing the explained variance, choose the number
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

#K Means
# Elbow method to find the optimal clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters =4 , init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
axes = plt.gca()
axes.set_xlim([-25,40])
axes.set_ylim([-10, 40])
plt.title('Clusters of customers')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

# Creating a DataFrame containing the timeseries data along with the pca features
data.iloc[:, 1:3] = X[:, 0:2]
ts = data.iloc[:, 0:3]
ts.columns = ['InvoiceDate', 'PCA1', 'PCA2']
ts.index = ts.InvoiceDate
ts = ts.iloc[:, 1:3]
#ts1 = ts.drop_duplicates()


UBO = ts

from pandas.tseries.offsets import CustomBusinessDay
week_mask = 'Mon'
bo1 = CustomBusinessDay(weekmask=week_mask)
BOM = ts.asfreq(freq=bo1, method='ffill')
# Applying PCA
from sklearn.decomposition import PCA
pca1 = PCA(n_components = 1)   # first run with the n_components=None , then after seeing the explained variance, choose the number
BOM = pca1.fit_transform(BOM)
explained_variance = pca1.explained_variance_ratio_
plt.plot(BOM, label='Monday')
plt.title('Biased Observer - Monday')
plt.legend('M')
plt.figure('1')


from pandas.tseries.offsets import CustomBusinessDay
week_mask = 'Tue'
bo2 = CustomBusinessDay(weekmask=week_mask)
BOT = ts.asfreq(freq=bo2, method='ffill')
# Applying PCA
from sklearn.decomposition import PCA
pca1 = PCA(n_components = 1)   # first run with the n_components=None , then after seeing the explained variance, choose the number
BOT = pca1.fit_transform(BOT)
explained_variance = pca1.explained_variance_ratio_
plt.plot(BOT)
plt.title('Biased Observer - Tuesday')
plt.legend('T')
plt.figure('2')

from pandas.tseries.offsets import CustomBusinessDay
week_mask = 'Wed'
bo3 = CustomBusinessDay(weekmask=week_mask)
BOW = ts.asfreq(freq=bo3, method='ffill')
# Applying PCA
from sklearn.decomposition import PCA
pca1 = PCA(n_components = 1)   # first run with the n_components=None , then after seeing the explained variance, choose the number
BOW = pca1.fit_transform(BOW)
explained_variance = pca1.explained_variance_ratio_
plt.plot(BOW)
plt.title('Biased Observer - Wednesday')
plt.legend('W')
plt.figure('3')

from pandas.tseries.offsets import CustomBusinessDay
week_mask = 'Thu'
bo4 = CustomBusinessDay(weekmask=week_mask)
BOTH = ts.asfreq(freq=bo4, method='ffill')
# Applying PCA
from sklearn.decomposition import PCA
pca1 = PCA(n_components = 1)   # first run with the n_components=None , then after seeing the explained variance, choose the number
BOTH = pca1.fit_transform(BOTH)
explained_variance = pca1.explained_variance_ratio_
plt.plot(BOTH)
plt.title('Biased Observer - Thursday')
plt.legend('T')
plt.figure('4')

from pandas.tseries.offsets import CustomBusinessDay
week_mask = 'Fri'
bo5 = CustomBusinessDay(weekmask=week_mask)
BOF = ts.asfreq(freq=bo5, method='ffill')
# Applying PCA
from sklearn.decomposition import PCA
pca1 = PCA(n_components = 1)   # first run with the n_components=None , then after seeing the explained variance, choose the number
BOF = pca1.fit_transform(BOF)
explained_variance = pca1.explained_variance_ratio_
plt.plot(BOF)
plt.title('Biased Observer - Friday')    
plt.legend('F')
plt.figure('5')