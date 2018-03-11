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
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r'''Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{'''
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            #Chunks everything
            #IN is preposition
            #DT is determiner
    except Exception as e:
        print(str(e))

process_content()

#Chinking is removal of something
#It is we have to chunk everything except for some things
