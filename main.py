import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("fake_or_real_news.csv")

data["false"] = data["label"].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)
x, y = data["text"], data["false"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vectorize = vectorizer.fit_transform(X_train)
X_test_vectorize = vectorizer.transform(X_test)

X_train_vectorize = X_train_vectorize.toarray()
X_test_vectorize = X_test_vectorize.toarray()

gnb = GaussianNB()
gnb.fit(X_train_vectorize, y_train)

y_pred = gnb.predict(X_test_vectorize)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
