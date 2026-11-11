import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load Data (If offline Excel/CSV given: df = pd.read_excel("Online Retail.xlsx"))
df = pd.read_csv("online_retail.csv")  # Change to your exam filename!
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 2. Simple RFM calculation without lambda functions
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["DaysAgo"] = (df["InvoiceDate"].max() - df["InvoiceDate"]).dt.days

cust = df.groupby("CustomerID").agg(
    Recency=("DaysAgo", "min"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
)
median_val = cust["Monetary"].median()   # compute once, not per row
segments = []
for value in cust["Monetary"]:
    if value > median_val:
        segments.append("High Value")
    else:
        segments.append("Low Value")
cust["Segment"] = segments 

X = cust[["Recency", "Frequency", "Monetary"]]
y = cust["Segment"]

# 3. Train & Plot ID3 Tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
id3 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42).fit(X_train, y_train)

print("Train Acc:", id3.score(X_train, y_train), "| Test Acc:", id3.score(X_test, y_test))
print("Importances:\n", pd.Series(id3.feature_importances_, index=X.columns))

plt.figure(figsize=(12, 6))
plot_tree(id3, feature_names=X.columns, class_names=id3.classes_, filled=True, fontsize=8)
plt.show()