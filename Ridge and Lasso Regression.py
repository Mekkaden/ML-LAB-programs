import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(X_train, y_train)
ridge = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=5).fit(X_train, y_train).best_estimator_
lasso = GridSearchCV(Lasso(max_iter=10000), {"alpha": [0.001, 0.01, 0.1, 1, 10]}, cv=5).fit(X_train, y_train).best_estimator_

for name, m in [("Linear", lin), ("Ridge", ridge), ("Lasso", lasso)]:
    pred = m.predict(X_test)
    print(name, "| MSE:", mean_squared_error(y_test, pred), "| R2:", r2_score(y_test, pred))
    plt.plot(m.coef_, "o-", label=name)

plt.legend();
plt.title("Coefficients Comparison"); 
plt.show()