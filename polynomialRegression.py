import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Offline CSV: df = pd.read_csv("mpg.csv").dropna()
df = sns.load_dataset("mpg")[["mpg", "displacement"]].dropna()
X = df[["displacement"]].values
y = df["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.scatter(X_test, y_test, alpha=0.4, label="Actual")

for d in [1, 2, 3, 4, 5]:
    poly = PolynomialFeatures(degree=d)
    X_tr_p = poly.fit_transform(X_train)
    X_te_p = poly.transform(X_test)
    
    model = LinearRegression().fit(X_tr_p, y_train)
    y_pred = model.predict(X_te_p)
    print(f"Degree {d} | MSE: {mean_squared_error(y_test, y_pred):.2f} | R2: {r2_score(y_test, y_pred):.4f}")
    
    if d in [1, 2, 4]:
        plt.plot(x_plot, model.predict(poly.transform(x_plot)), label=f"Degree {d}")

plt.legend();
plt.title("Polynomial Regression");
plt.show()