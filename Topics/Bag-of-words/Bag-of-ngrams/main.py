from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2, 3))
text = sent_tokenize(open('data/dataset/input.txt', 'r').read())

X = vectorizer.fit_transform(text)

print(list(vectorizer.get_feature_names_out()))
