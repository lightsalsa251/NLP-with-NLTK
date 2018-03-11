# NLP is converting spoken language to numbers
#tokenizing -> Grouping things(word tokenizers and sentence tokenizers)
#corpora -> Body of text
#lexicons -> Words and their meanings like a dictionary but some words can have different contrasting meanings

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

sample = 'Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish blue and you should not eat cardboard.'
#tokenize will eliminate working with re

print(sent_tokenize(sample))
print(word_tokenize(sample))

#keeps every . which is a part of the word
#alt+3 for commenting multiple lines or a single line

for i in word_tokenize(sample):
    print(i)

#nltk works with other languages too
