import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load YOUR specific CSV
df = pd.read_csv('diabetes_prediction_dataset.csv')
y = df['diabetes']

X = df.drop('diabetes', axis=1)
X = pd.get_dummies(X, drop_first=True, dtype=int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

for name, X_tr, X_te in [("Without Scaling", X_train, X_test), ("With Scaling", X_train_s, X_test_s)]:
    # max_iter increased because this dataset is much larger
    m = LogisticRegression(max_iter=2000).fit(X_tr, y_train)
    pred = m.predict(X_te)
    
    print(f"{name:>15} | Acc: {accuracy_score(y_test, pred):.4f} | Prec: {precision_score(y_test, pred):.4f} | Rec: {recall_score(y_test, pred):.4f} | F1: {f1_score(y_test, pred):.4f}")