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

            namedEnt = nltk.ne_chunk(tagged,binary=True)
            #binary = True classifies everything as a named entity period and without this it gives the different types
            #namedEnt = nltk.ne_chunk(tagged)
            #false positives and errors are high in name entity than in nouns but both can be used with respect to a application
            namedEnt.draw()
    except Exception as e:
        print(str(e))
'''
There are several NE Type keywords like GPE,Person etc
'''
process_content()
