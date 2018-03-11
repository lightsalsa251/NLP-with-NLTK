#when we have tokenized and used stemming the next part to figure out the meaning is chunking
#when there are multiple nouns then there is a problem to whom is the sentence for
#In chunking there will be a noun with its descriptive phase present in that sentence
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

            chunkGram = r'''Chunk: {<RB.?>*<VB.*>*<NNP>+<NN>?}'''
            #Putting r in front means consider it as raw5
            #Any adverb but 0 or 1 length
            #NNP is a proper noun
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            chunked.draw()
    except Exception as e:
        print(str(e))
#RB is adverb
process_content()
