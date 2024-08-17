import string

from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from math import sqrt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

file = open('news.xml', 'r').read()
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
soup = BeautifulSoup(file, 'xml')

news = soup.find_all('news')
extra_weight = 3.0

def print_header(header: str) -> None:
    print(f'HEADER: {header}')
def print_text(sentences: list) -> None:
    print('TEXT:', '\n'.join([f'{x}' for x in sentences]), '\n')


def tokenize_words(sentence: str) -> list:
    lemmatizer = WordNetLemmatizer()
    tokens = [i.lower() for i in word_tokenize(sentence) if i.lower() not in stopwords.words('english')]
    tokens = list(filter(lambda token: token not in string.punctuation, tokens))

    return(' ').join(lemmatizer.lemmatize(token, pos='n') for token in tokens)

# news_list = list()
for new in news:
    #Tokenize sentences in  <value name="text"> nodes
    tmp = sent_tokenize(new.find('value', {'name': 'text'}).text)
    #extract header nodes
    header = new.find('value', {'name': 'head'}).text
    #Find √N (N is the number of the source text's sentences).
    n_lines = round(sqrt(len(tmp)))
    #preprocess headers
    tokenized_header = tokenize_words(header)

    #preprocess sentences
    tokenized_sent = [tokenize_words(x) for x in tmp]
    # news_list.append({'header': header,
    #                   'text': tokenized_sent
    #                   })

    #Calculate TF-IDF of each documents
    X = vectorizer.fit_transform(tokenized_sent)

    #Find and add extra weights
    for word in tokenized_header.split(' '):
        try:
            word_index = vectorizer.vocabulary_[word]
            if word_index in X.indices:
                X[:, word_index] *= extra_weight
        except KeyError:
            continue

    dtype = [('index', int), ('mean', float)]
    mean_arr = np.array([np.mean(X[i].data) for i in range(X.shape[0])], dtype=dtype) #calculate mean TF-IDF for document
    sorted_array = np.sort(np.argsort(mean_arr, order='mean')[::-1][:n_lines]) #get get the best n indexes ordered by mean value

    print_header(header)
    print_text(np.array(tmp)[sorted_array])


