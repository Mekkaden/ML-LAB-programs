import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score ,f1_score
from sklearn.neighbors import KNeighborsClassifier
import time

df = pd.read_csv("/content/fashion-mnist_train.csv")
x = df.drop("label",axis=1)
x = x.iloc[:6000,:] / 255.0
y = df["label"][:6000].astype(int)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42,stratify = y)

accs = []

for k in [1,3,5,7]:
  m = KNeighborsClassifier(n_neighbors=k)
  stime = time.time()
  acc = accuracy_score(ytest,m.fit(xtrain,ytrain).predict(xtest))
  accs.append(acc)
  print(f"accuracy is {acc} when k is {k} , time taken is {time.time() - stime}")

plt.plot([1,3,5,7] , accs)
plt.show()


#To implement a K-Nearest Neighbors (KNN) classifier for image classification on
# the Fashion MNIST dataset, evaluate its classification accuracy
# across different values of $k$ ($k = 1, 3, 5, 7$), and plot an accuracy vs.
# $k$ graph to determine the optimal neighbor value.










