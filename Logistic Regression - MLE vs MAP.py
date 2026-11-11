import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('breast-cancer.csv')

y_raw = df['diagnosis']


y = (df['diagnosis'] == 'M').astype(int).values

X = df.drop(['id', 'diagnosis'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

mle = LogisticRegression(C=1e10, max_iter=5000).fit(X_train, y_train)
map_l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=5000).fit(X_train, y_train)
map_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000).fit(X_train, y_train)

for name, m in [("MLE (No Prior)", mle), ("MAP (L2)", map_l2), ("MAP (L1)", map_l1)]:
    pred = m.predict(X_test)
    print(name, "| Acc:", accuracy_score(y_test, pred), "| F1:", f1_score(y_test, pred), "| Norm:", np.linalg.norm(m.coef_))