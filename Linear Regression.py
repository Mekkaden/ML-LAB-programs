import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
df = fetch_california_housing(as_frame=True).frame
X = df[["AveRooms"]].values
y = df["MedHouseVal"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Prepend column of 1s (Bias)
X_b_train = np.c_[np.ones(len(X_train)), X_train]
X_b_test = np.c_[np.ones(len(X_test)), X_test]

# 2. Gradient Descent
theta_gd = np.zeros(2)
costs = []

for i in range(1000):
    error = X_b_train.dot(theta_gd) - y_train
    costs.append(np.mean(error ** 2) / 2)
    theta_gd = theta_gd - 0.01 * (1 / len(y_train)) * X_b_train.T.dot(error)

y_gd = X_b_test.dot(theta_gd)

# 3. Normal Equation
theta_ne = np.linalg.inv(X_b_train.T.dot(X_b_train)).dot(X_b_train.T).dot(y_train)
y_ne = X_b_test.dot(theta_ne)

# 4. Scikit-Learn
sk = LinearRegression().fit(X_train, y_train)
y_sk = sk.predict(X_test)

# 5. Output Table
for name, pred in [
    ("Gradient Descent", y_gd),
    ("Normal Equation", y_ne),
    ("scikit-learn", y_sk)
]:
    print(f"{name} | MSE: {mean_squared_error(y_test, pred)} | R2: {r2_score(y_test, pred)}")

# 6. Gradient Descent Convergence Plot
plt.plot(costs)
plt.title("Gradient Descent Convergence")
plt.show()

# 7. Actual vs Predicted Plot
plt.scatter(X_test, y_test, alpha=0.3, label="Actual")
plt.plot(X_test, y_gd, color="red", label="GD")
plt.plot(X_test, y_ne, color="brown", linestyle="--", label="Normal Eq")
plt.plot(X_test, y_sk, color="green", linestyle=":", label="Sklearn")
plt.legend()
plt.show()
