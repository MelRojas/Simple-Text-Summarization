from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
text = sent_tokenize(open('data/dataset/input.txt', 'r').read())

X = vectorizer.fit_transform(text)
print(X.toarray())


