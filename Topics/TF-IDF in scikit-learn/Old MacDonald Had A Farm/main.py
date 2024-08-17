from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(input='file', use_idf=True, lowercase=True,
                             analyzer='word', ngram_range=(1, 1),
                             stop_words=None)
weighted_matrix = vectorizer.fit_transform([open('data/dataset/input.txt', 'r')])

print(weighted_matrix.toarray()[0][10])
