from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("news.csv");
X_text = df["text"];
y = df["target"]
X_tr_txt, X_te_txt, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42,stratify=news.target)

vec = CountVectorizer(stop_words="english", max_features=5000)
X_train = vec.fit_transform(X_tr_txt)
X_test = vec.transform(X_te_txt)

for name, model in [("Multinomial NB", MultinomialNB()), ("Bernoulli NB", BernoulliNB())]:
    pred = model.fit(X_train, y_train).predict(X_test)
    print(name, "| Acc:", accuracy_score(y_test, pred), "| F1:", f1_score(y_test, pred, average="macro"))