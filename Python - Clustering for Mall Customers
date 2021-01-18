#%% Kaggle data - Mall Customer Segmentation


from datetime import date
import sys

today = date.today()
print("As of:", today)
print("Author: ", "Jake Chen")
print("Python version:", sys.version)
print("Link: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python")

#%%         Import libraries
#---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%         Import data
#---------------------------------------------------------------

import os
os.chdir("C:\\Users\\dimen\\Documents\\Python\\Kaggle - Mall Customer Segmentation\\")
df = pd.read_csv("Mall_Customers.csv")


#%%         Inspect data
#---------------------------------------------------------------

df.info()
df.describe()
df.head()

#%%         Handling missing values
#---------------------------------------------------------------

df.isnull().sum() #count of missing values in each column
df.isnull().sum() /len(df)*100 #percentage of missing values in each column

df.isnull().any().sum() #Number of columns with missing values
df.isnull().any(axis=1).sum() #Number of rows with missing values

#%%         Handling categorical values
#---------------------------------------------------------------

#Binary encoding
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})


#%%         Scaling data
#---------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaled_features = scaler.fit_transform(df[num_features])
#df[num_features] = scaler.fit_transform(df[num_features])

features = df

#%%         PCA (Principal Component Analysis)
#---------------------------------------------------------------

#Transforming the numeric features to 2 PCs for ease of visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)

#Visualize the 2 PCs
plt.scatter(features_2d[:, 0], features_2d[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show()


#%%         WCSS (Within cluster sum of squares)
#---------------------------------------------------------------
#WCSS helps determine the optimal number of clusters - lower means data points are closer

from sklearn.cluster import KMeans

# Create 10 models with 1 to 10 clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    # Fit the data points
    kmeans.fit(features.values)
    # Get the WCSS (inertia) value
    wcss.append(kmeans.inertia_)

# Plot the WCSS values onto a line graph
plt.plot(range(1, 11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#The optimal point is at the 'elbow' (after this point, the reduction in WCSS is less pronounced)


#%%         K-means clustering
#---------------------------------------------------------------
from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
km_clusters

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color=colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)

#%%         Hierarchical clustering (agglomerative clustering)
#---------------------------------------------------------------
#Agglomerative clustering is a "bottom up** approach

from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
agg_clusters

import matplotlib.pyplot as plt

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, agg_clusters)


