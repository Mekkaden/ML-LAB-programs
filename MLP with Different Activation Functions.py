import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Offline CSV: df = pd.read_csv("mnist.csv");
# X = df.drop("label", axis=1).values;
# y = df["label"].values
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X, y = X / 255.0, y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=6000, test_size=1500, random_state=42)

for act in ["logistic", "relu", "tanh"]:
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation=act, max_iter=100, random_state=42).fit(X_train, y_train)
    pred = mlp.predict(X_test)
    print(f"Act={act} | Acc: {accuracy_score(y_test, pred):.4f} | Iters: {mlp.n_iter_} | Time: {time.time() - t0:.2f}s")