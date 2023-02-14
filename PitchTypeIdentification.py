#PitchTypeIdentification.py
# Using a k-means clustering algorithm to take in pitch data and assign a pitch type

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy.linalg import norm
import pandas as pand
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

df = pand.read_excel('Live_Feb12.xlsx', usecols="N:O", skiprows=4)

def plot_pitches(dataframe, title):
    groups = dataframe.groupby('Pitch Type')
    
    for name, group in groups:
        x = group['HB (trajectory)']
        y = group['VB (trajectory)']
        locs = pand.DataFrame([x,y])
        plt.scatter(x, y, label=name)

    plt.axis('equal')
    plt.legend()
    plt.title(title)
    plt.show()

#plot_pitches(df, "Raw Pitch Break Data Visualization")

X_std = StandardScaler().fit_transform(df)
km = KMeans(n_clusters=4, max_iter=500)
km.fit(X_std)
centroids = km.cluster_centers_

fig, ax = plt.subplots(figsize = (6,6))
plt.scatter(X_std[km.labels_ == 0, 0], X_std[km.labels_ == 0, 1], c = 'green', label = 'cluster 1')
plt.scatter(X_std[km.labels_ == 1, 0], X_std[km.labels_ == 1, 1], c = 'blue', label = 'cluster 2')
plt.scatter(X_std[km.labels_ == 2, 0], X_std[km.labels_ == 2, 1], c = 'grey', label = 'cluster 3')
plt.scatter(X_std[km.labels_ == 3, 0], X_std[km.labels_ == 3, 1], c = 'black', label = 'cluster 4')
plt.scatter(centroids[:, 0], centroids[:, 1], marker = '*', s = 300, c = 'r', label = 'centroid')

plt.legend()
plt.xlabel('Normalized horizontal break (trajectory)')
plt.ylabel('Normalized vertical break (trajectory)')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal')
plt.show()
