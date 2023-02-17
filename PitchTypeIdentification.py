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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

pitch_data = pand.read_excel('Live_Feb12.xlsx', usecols="B:U")

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

#plot_pitches(pitch_data, "Raw Pitch Break Data Visualization")

def pitch_type_clustering_three(pitch_data):
    three_feature = pitch_data.filter(['Velocity', 'HB (trajectory)', 'VB (trajectory)'])
    km_three = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    y = km_three.fit_predict(three_feature)
    print(y)

    pitch_data['Cluster'] = y
    print(pitch_data)

    c1 = pitch_data[pitch_data.Cluster == 0]
    c2 = pitch_data[pitch_data.Cluster == 1]
    c3 = pitch_data[pitch_data.Cluster == 2]
    c4 = pitch_data[pitch_data.Cluster == 3]

    kplot = plt.axes(projection='3d')
    xline = np.linspace(0, 15, 1000)
    yline = np.linspace(0, 15, 1000)
    zline = np.linspace(0, 15, 1000)
    kplot.plot3D(xline, yline, zline, 'black')
    # Data for three-dimensional scattered points
    kplot.scatter3D(c1['Velocity'], c1['HB (trajectory)'], c1['VB (trajectory)'], c='red', label = 'Cluster 1')
    kplot.scatter3D(c2['Velocity'], c2['HB (trajectory)'], c2['VB (trajectory)'],c ='green', label = 'Cluster 2')
    kplot.scatter3D(c3['Velocity'], c3['HB (trajectory)'], c3['VB (trajectory)'], c='blue', label = 'Cluster 3')
    kplot.scatter3D(c4['Velocity'], c4['HB (trajectory)'], c4['VB (trajectory)'],c ='black', label = 'Cluster 4')
    plt.scatter(km_three.cluster_centers_[:,0], km_three.cluster_centers_[:,1], color = 'indigo', s = 200)
    plt.legend()
    plt.xlabel('Velocity')
    plt.ylabel('Normalized horizontal break (trajectory)')
    plt.clabel('Normalized vertical break (trajectory)')
    plt.title("Pitch Type Clusters")
    plt.show()

class Pitch_Type_Class:

    def __init__(self, pitch_data):
        self.pitch_data = pitch_data 
        X = pitch_data.drop(columns = ['Pitch Type'])
        Y = pitch_data['Pitch Type'].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X_train, Y_train)

    def pitch_type_knn(self, pitch_data):
        X = pitch_data.drop(columns = ['Pitch Type'])
        Y = pitch_data['Pitch Type'].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, Y_train)
        print(X_test)
        print(knn.predict(X_test))
        print(knn.score(X_test, Y_test))
    
    def predict(self, dataset):
        pitch_types = self.knn.predict(dataset)
        dataset['Pitch Type'] = pitch_types.tolist()
        print(dataset)


data = pitch_data.filter(['Pitch Type', 'Velocity', 'HB (trajectory)', 'VB (trajectory)'])
pitch_class = Pitch_Type_Class(data)
#pitch_class.pitch_type_knn(data)
pitch_class.predict(pitch_data.filter(['Velocity', 'HB (trajectory)', 'VB (trajectory)']))

