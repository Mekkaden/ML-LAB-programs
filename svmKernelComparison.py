import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Offline CSV: df = pd.read_csv("fashion.csv"); X = df.drop("label", axis=1).values; y = df["label"].values
X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
X = X[:4000] / 255.0
y = y[:4000].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

for k in ["linear", "poly", "rbf"]:
    t0 = time.time()
    clf = SVC(kernel=k).fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f"Kernel={k} | Acc: {accuracy_score(y_test, pred):.4f} | Time: {time.time() - t0:.2f}s")
    
#To implement and compare the performance of Support Vector Machine (SVM) classifiers using different kernel
# functions (Linear, Polynomial, and Radial Basis Function / RBF) on the Fashion-MNIST dataset in terms
# of classification accuracy and training/execution time.