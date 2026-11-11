import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Offline CSV fallback: texts = pd.read_csv("news.csv")["text"]
data = fetch_20newsgroups(subset="train", categories=["sci.space"])
vec = CountVectorizer(max_features=2000, stop_words="english")
X = vec.fit_transform(data.data).toarray()

#counts the number of occurences acros sthe document
counts = X.sum(axis=0)

#counts totoal numeb of words across doc
N = counts.sum()

#vocabulary size 
V = len(counts)


#probablility. = number of favourbale items by totoal numbe of items 
theta_mle = counts / N

for a in [0.1, 1, 5]:
    theta_map = (counts + a) / (N + a * V)
    print(f"alpha={a} | Mean Diff: {np.abs(theta_mle - theta_map).mean():.6f} | Max Prob: {theta_map.max():.6f}")

top10 = np.argsort(theta_mle)[::-1][:10]
print("Top 10 words:", vec.get_feature_names_out()[top10])


#The aim of this code is to estimate word probability distributions over a text 
# by comparing Maximum Likelihood Estimation (MLE) against Maximum A Posteriori (MAP) 
# estimation using Laplace/Lidstone smoothing across different values of hyperparameter 
#α. It analyzes how additive Dirichlet smoothing pulls probability mass toward a uniform
# prior (evaluated via mean difference and peak probability) and extracts the 10 most frequent
# space-related vocabulary words.#