import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
ag = AgglomerativeClustering(n_clusters=5).fit(X)

print("KMeans | Inertia:", km.inertia_, "| Silhouette:", silhouette_score(X, km.labels_))
print("Agglomerative | Silhouette:", silhouette_score(X, ag.labels_))


#**************
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=km.labels_);
plt.title("K-Means")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=ag.labels_);
plt.title("Agglomerative")

plt.show()

dendrogram(linkage(X, method="ward"))
plt.title("Dendrogram");
plt.show()


#The aim of this code is to segment mall customers into distinct groups based on their annual income
# and spending score using both K-Means and Agglomerative Hierarchical Clustering. 
# It compares the two algorithms quantitatively via silhouette scores and inertia, 
# while visually evaluating them by plotting side-by-side scatter plots 
# (Annual Income vs. Spending Score colored by cluster assignments) and a hierarchical tree dendrogram using Ward's linkage.