import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# nltk.download('stopwords')

sw = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=sw)
text = sent_tokenize(open('data/dataset/input.txt', 'r').read())

X = vectorizer.fit_transform(text)

print(X.toarray())
