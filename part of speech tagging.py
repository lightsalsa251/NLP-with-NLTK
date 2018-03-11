import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#PunktSentenceTokenizer is an unsupervised ml tokenizer
#It comes trained and it can be retrained too

train = state_union.raw('2005-GWBush.txt')
sample = state_union.raw('2006-GWBush.txt')

#training and testing data

custom_sent_tokenizer = PunktSentenceTokenizer(train)
tokenized = custom_sent_tokenizer.tokenize(sample)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process_content()

#pos creates tupples of each word and its tag

'''
POS tag list
CC  coordinating conjuction
CD cardinal digit
DT determiner
and many more
'''
