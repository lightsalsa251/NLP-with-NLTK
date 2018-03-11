#most of the work is organising, structuring and pre-processing data
#the stem of riding is rid
#words can have different affixes to them but their meaning is the same like ride and riding
#reduces redundancy
# porter stemming algorithm is used

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

sample = ['python','pythoner','pythoning','pythoned','pythonly']

for w in sample:
    print(ps.stem(w))

new_text = 'It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once.'

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))

#Stemming is not used much, WordNet is used more or ImageNet
