import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(open('./data/dataset/input.txt', 'r').read())

for token in doc:
    print(token.lemma_, end=' ')