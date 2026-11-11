import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = load_digits(return_X_y=True)

ks = range(2, 15)
inertias, sils = [], []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, km.labels_))

print("Best K by silhouette:", list(ks)[np.argmax(sils)])

plt.subplot(1, 2, 1)
plt.plot(ks, inertias, "o-"); 
plt.title("Elbow Method")

plt.subplot(1, 2, 2)
plt.plot(ks, sils, "o-", color="green");
plt.title("Silhouette Score")
plt.show()


#The aim of this code is to find the optimal number of clusters ($k$) for grouping handwritten digit images using K-Means clustering.
# It tests cluster counts from 2 to 14, evaluates clustering performance using inertia and silhouette scores,
# and visualizes the results side-by-side using the Elbow Method and a Silhouette Score curve to determine the best $k$.