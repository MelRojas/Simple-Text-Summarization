from nltk.stem import LancasterStemmer


# the following line reads a text from the input and converts it into a list
sent = input().split()
lancaster = LancasterStemmer()

# write your code here
for item in sent:
    print(lancaster.stem(item))