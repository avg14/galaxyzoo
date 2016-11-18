import pickle
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

with open('transformed_set.out', 'r') as f:
	X = pickle.load(f)

kmeans_model = KMeans(n_clusters=2, random_state=1).fit(X)
labels = kmeans_model.labels_
print "Kmeans score:", metrics.silhouette_score(X, labels, metric='euclidean')

agg_model = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)
labels = agg_model.labels_
print "Agglomerative (Ward) score:", metrics.silhouette_score(X, labels, metric='euclidean')

agg_model = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(X)
labels = agg_model.labels_
print "Agglomerative (Complete) score:", metrics.silhouette_score(X, labels, metric='euclidean')
