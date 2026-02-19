import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#Generate synthetic data and labels
features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=None #42 is used in example, best practice is to set to None
)

#Feature scaling using standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#Instantiate the Kmeans class
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10, #number of kmeans runs to preform. Run with lowest SSE value is returned
    max_iter=300, #max number of iterations per run
    random_state=42
)

#run the kmeans clustering
kmeans.fit(scaled_features)

#The lowest SSE value
kmeans.inertia_


#Final locations of the centroid
kmeans.cluster_centers_


#The number of iterations required to converge
kmeans.n_iter_

#first five predicted labels
kmeans.labels_[:5]