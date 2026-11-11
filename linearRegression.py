import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # ADDED THIS

# 1. Load Data (Offline CSV: df = pd.read_csv("housing.csv"))
df = fetch_california_housing(as_frame=True).frame
X = df[["AveRooms"]].values
y = df["MedHouseVal"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ADDED SCALING HERE ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# --------------------------

# prepend column of 1s (bias) for scratch methods (UPDATED to use scaled data)
X_b_train = np.c_[np.ones(len(X_train_scaled)), X_train_scaled]
X_b_test = np.c_[np.ones(len(X_test_scaled)), X_test_scaled]

# 2. Gradient Descent (Inline - no functions)
theta_gd = np.zeros(2)
costs = []
for i in range(1000):
    error = X_b_train.dot(theta_gd) - y_train
    costs.append(np.mean(error ** 2) / 2)
    theta_gd = theta_gd - 0.01 * (1 / len(y_train)) * X_b_train.T.dot(error)
y_gd = X_b_test.dot(theta_gd)

# 3. Normal Equation (1 line)
theta_ne = np.linalg.inv(X_b_train.T.dot(X_b_train)).dot(X_b_train.T).dot(y_train)
y_ne = X_b_test.dot(theta_ne)

# 4. Scikit-Learn (UPDATED to use scaled data)
#*****************************************************
sk = LinearRegression().fit(X_train_scaled, y_train)
y_sk = sk.predict(X_test_scaled)

# 5. Output Table & Plots
for name, pred in [("Gradient Descent", y_gd), ("Normal Equation", y_ne), ("scikit-learn", y_sk)]:
    print(f"{name} | MSE: {mean_squared_error(y_test, pred)} | R2: {r2_score(y_test, pred)}")

plt.plot(costs)
plt.title("Gradient Descent Convergence"); 
plt.show()

plt.scatter(X_test, y_test, alpha=0.3, label="Actual")
plt.plot(X_test, y_gd, color="red", label="GD")
plt.plot(X_test, y_ne, color="brown", linestyle="--", label="Normal Eq")
plt.plot(X_test, y_sk, color="green", linestyle=":", label="Sklearn")
plt.legend();
plt.show()