#stop words can be sarcastic words and those words which can be ignored
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sample = 'This is an example showing of stop word filteration.'
stop_words = set(stopwords.words('english'))
#print(stop_words)
words = word_tokenize(sample)

##filtered_sentence = []
##
##for w in words:
##    if w not in stop_words:
##        filtered_sentence.append(w)

filtered_sentence = [w for w in words if not w in stop_words]
        
print(filtered_sentence)
