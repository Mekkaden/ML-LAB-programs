# Machine Learning Lab Programs

A curated collection of core Machine Learning lab implementations and comparative experiments implemented in Python. This repository covers supervised learning (regression and classification), unsupervised clustering, probabilistic modeling, and neural networks using both scratch implementations and standard libraries (`scikit-learn`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`).

---

## 📋 Program Index

| # | Program File | Topic / Domain | Model / Algorithm | Dataset |
|---|---|---|---|---|
| 01 | [`Linear Regression.py`](<./Linear Regression.py>) | Regression | Gradient Descent (Scratch), Normal Equation (Scratch), Scikit-Learn | California Housing |
| 02 | [`Linear SVM.py`](<./Linear SVM.py>) | Classification | Linear Support Vector Machine (Decision Boundary & Margins) | Iris Dataset (Setosa vs Non-Setosa) |
| 03 | [`Polynomial Regression.py`](<./Polynomial Regression.py>) | Regression | Polynomial Regression (Degrees 1 to 5) | Auto MPG (`displacement` vs `mpg`) |
| 04 | [`Ridge and Lasso Regression.py`](<./Ridge and Lasso Regression.py>) | Regularized Regression | Linear vs Ridge vs Lasso with `GridSearchCV` | Diabetes Dataset |
| 05 | [`Logistic Regression with Scaling.py`](<./Logistic Regression with Scaling.py>) | Binary Classification | Logistic Regression with & without `StandardScaler` | Diabetes Prediction Dataset |
| 06 | [`Logistic Regression - MLE vs MAP.py`](<./Logistic Regression - MLE vs MAP.py>) | Probabilistic Classification | Logistic Regression: MLE (No Prior), MAP ($L_2$), MAP ($L_1$) | Breast Cancer Dataset |
| 07 | [`SVM Kernel Comparison.py`](<./SVM Kernel Comparison.py>) | Classification | SVM Kernel Comparison (Linear vs Polynomial vs RBF) | Fashion-MNIST |
| 08 | [`KNN Classifier.py`](<./KNN Classifier.py>) | Classification | K-Nearest Neighbors ($k \in \{1, 3, 5, 7\}$) & Accuracy vs $k$ Plot | Fashion-MNIST |
| 09 | [`Naive Bayes Classifier.py`](<./Naive Bayes Classifier.py>) | Text Classification | Multinomial Naive Bayes vs Bernoulli Naive Bayes | 20 Newsgroups (`news.csv`) |
| 10 | [`Multinomial Naive Bayes - MLE vs MAP.py`](<./Multinomial Naive Bayes - MLE vs MAP.py>) | Probability Estimation | Maximum Likelihood (MLE) vs MAP (Dirichlet/Laplace Smoothing) | 20 Newsgroups (`sci.space`) |
| 11 | [`K-Means Clustering.py`](<./K-Means Clustering.py>) | Unsupervised Clustering | K-Means with Elbow Method & Silhouette Analysis ($k=2..14$) | Optical Recognition of Handwritten Digits |
| 12 | [`Hierarchical vs K-Means.py`](<./Hierarchical vs K-Means.py>) | Unsupervised Clustering | Agglomerative Hierarchical Clustering vs K-Means & Dendrogram | Mall Customers Dataset |
| 13 | [`ID3 Decision Tree.py`](<./ID3 Decision Tree.py>) | Decision Trees | ID3 Decision Tree (`criterion="entropy"`) with RFM Segmentation | Online Retail Dataset |
| 14 | [`MLP with Different Layers.py`](<./MLP with Different Layers.py>) | Neural Networks | Multilayer Perceptron Architecture Comparison (1, 2, 3 Layers) | Red Wine Quality Dataset |
| 15 | [`MLP with Different Activation Functions.py`](<./MLP with Different Activation Functions.py>) | Neural Networks | MLP Classifier Activation Comparison (`logistic`, `relu`, `tanh`) | MNIST 784 Dataset |

---

## 🔍 Module Descriptions

### 1. Regression Models

- **`Linear Regression.py`**: Compares three approaches to fit a linear model on California housing data:
  1. Gradient Descent implemented from scratch with cost tracking across iterations.
  2. The analytical Normal Equation $(X^T X)^{-1} X^T y$.
  3. `scikit-learn`'s `LinearRegression`.
- **`Polynomial Regression.py`**: Explores polynomial curve fitting on vehicle engine displacement vs MPG across polynomial degrees $1$ through $5$, evaluating bias, variance, and overfitting.
- **`Ridge and Lasso Regression.py`**: Demonstrates $L_2$ (Ridge) and $L_1$ (Lasso) penalty regularization with 5-fold cross-validated hyperparameter tuning (`GridSearchCV`) over penalty weights $\alpha$, plotting resulting model weights/coefficients.

### 2. Classification Models

- **`Linear SVM.py`**: Trains a linear Support Vector Classifier on Iris sepal dimensions and visualizes the 2D decision boundary hyperplane, safety margins, and highlighted support vectors.
- **`SVM Kernel Comparison.py`**: Benchmarks Linear, Polynomial, and RBF (Radial Basis Function) kernels on high-dimensional image data (Fashion-MNIST), measuring accuracy and runtime latency.
- **`Logistic Regression with Scaling.py`**: Evaluates the effect of feature scaling (`StandardScaler`) on logistic regression performance (accuracy, precision, recall, and F1 score) on tabular patient data.
- **`Logistic Regression - MLE vs MAP.py`**: Implements unregularized Maximum Likelihood Estimation ($C \to \infty$) alongside Maximum A Posteriori with Gaussian prior ($L_2$) and Laplace prior ($L_1$, sparsity-inducing) on breast cancer diagnostics.
- **`KNN Classifier.py`**: Explores instance-based learning with $k$-NN over neighborhood sizes $k \in \{1, 3, 5, 7\}$, reporting training/inference times and plotting accuracy vs $k$.
- **`Naive Bayes Classifier.py`**: Compares word-frequency (`MultinomialNB`) vs binary term-presence (`BernoulliNB`) formulations for text categorization.

### 3. Probabilistic & Decision Trees

- **`Multinomial Naive Bayes - MLE vs MAP.py`**: Formulates raw term-frequency probability estimation (MLE) versus smoothed estimation (MAP) using Laplace/Lidstone smoothing hyperparameter $\alpha \in \{0.1, 1, 5\}$, extracting the top informative space terms.
- **`ID3 Decision Tree.py`**: Performs Recency, Frequency, and Monetary (RFM) customer value segmentation, then trains and visualizes an ID3-style entropy-driven decision tree (`DecisionTreeClassifier(criterion="entropy")`).

### 4. Clustering & Unsupervised Learning

- **`K-Means Clustering.py`**: Finds optimal cluster counts using both the Elbow Method (sum of squared cluster distances / inertia) and Silhouette Score analysis across $k \in [2, 14]$.
- **`Hierarchical vs K-Means.py`**: Evaluates customer grouping via K-Means and Ward's hierarchical agglomerative clustering, rendering side-by-side cluster visualizations and a linkage dendrogram.

### 5. Neural Networks (MLP)

- **`MLP with Different Layers.py`**: Evaluates Multi-Layer Perceptron architectures comparing 1-layer `(16,)`, 2-layer `(32, 16)`, and 3-layer `(64, 32, 16)` hidden layers to predict wine quality classes.
- **`MLP with Different Activation Functions.py`**: Analyzes the convergence speed, iteration count, and test accuracy of Sigmoid (`logistic`), Rectified Linear Unit (`relu`), and Hyperbolic Tangent (`tanh`) activations on MNIST digits.

---

## 🚀 Getting Started

### Prerequisites

Python 3.8 or higher is recommended. Install all dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Running a Program

Run any script directly with python:

```bash
python "Linear Regression.py"
python "Linear SVM.py"
python "MLP with Different Layers.py"
python "K-Means Clustering.py"
```

> **Note on Datasets:**  
> Several programs fetch datasets directly online via `scikit-learn` or `seaborn` (e.g. California Housing, Iris, Digits, Fashion-MNIST, MNIST 784, 20 Newsgroups). For programs loading local CSV files (e.g., `Mall_Customers.csv`, `online_retail.csv`, `breast-cancer.csv`, `diabetes_prediction_dataset.csv`, `news.csv`), ensure the respective CSV file is located in the working directory.

---

## ☕ Support

Buy me a coffee when you see me ☕
