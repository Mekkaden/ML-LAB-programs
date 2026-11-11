#The aim of this code is to load and scale the red wine dataset, train three different neural network 
# (1-layer, 2-layer, and 3-layer), and compare their accuracy and F1 scores to see which network structure best predicts wine quality.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# Offline CSV: df = pd.read_csv("winequality-red.csv") (Check if sep=";" or sep=",")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

for layers in [(16,), (32, 16), (64, 32, 16)]:
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=500, random_state=42).fit(X_train, y_train)
    pred = mlp.predict(X_test)
    print(f"Layers {layers} | Acc: {accuracy_score(y_test, pred):.4f} | F1: {f1_score(y_test, pred, average='macro'):.4f}")