import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- Data Loading and Model Training (Your original code) ---
df = load_iris(as_frame=True).frame
x = df[["sepal length (cm)", "sepal width (cm)"]]
y = (df["target"] == 0).astype(int).values

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
svm = SVC(C=1.0, kernel="linear").fit(xtrain, ytrain)
print(f"test accuracy = {svm.score(xtest, ytest)}")

# --- Corrected Plotting Section ---
# 1. Correctly generate the x-axis range using .iloc for the first column
x_min, x_max = x.iloc[:, 0].min(), x.iloc[:, 0].max()
x_plot = np.linspace(x_min, x_max, 300)

# 2. Extract weights and intercept
w = svm.coef_[0]
b = svm.intercept_[0]

# 3. Calculate the decision boundary lines
y_bound = -(w[0] * x_plot + b) / w[1]
y_up = -(w[0] * x_plot + b - 1) / w[1]
y_down = -(w[0] * x_plot + b + 1) / w[1]

# 4. Plotting
plt.figure(figsize=(8, 6))

# Scatter the original data points (using .iloc for positional access)
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# Plot the decision boundary and margins
plt.plot(x_plot, y_bound, 'k-', label='Decision Boundary')
plt.plot(x_plot, y_up, 'k--', label='Margin')
plt.plot(x_plot, y_down, 'k--')

# Highlight the support vectors
plt.scatter(
    svm.support_vectors_[:, 0],
    svm.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors='none',
    edgecolors='k',
    label='Support Vectors'
)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.title('SVM Linear Decision Boundary (Iris Dataset)')
plt.show()


#To implement a Linear Support Vector Machine (SVM) binary classifier on the Iris dataset to classify
# Iris-Setosa versus Non-Setosa based on sepal length and sepal width,
# evaluate its classification accuracy, and visualize the optimal hyperplane, margins, and support vectors.

#df = pd.read_csv("/content/Iris.csv")
#x = df.iloc[:,1:3]
#y=(df["Species"]==("Iris-setosa")).astype(int).values





