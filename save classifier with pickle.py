import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
##documents can be used for training
##documents = []
##for category in movie_reviews.categories():
##    for fileid in movie_reviews.fileids(category):
##        documents.append({list(movie_reviews.words(fileid)),category])
random.shuffle(documents)
#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
#creates bag of words

all_words = nltk.FreqDist(all_words)
##print(all_words.most_common(15))
##print(all_words['stupid'])    

word_features = list(all_words.keys())[:3000]
#top 3000 words

def find_features(documents):
    words = set(documents)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    #if w is in documents
    return features
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev,category) in documents]

training_set = featuresets[:1900]
testing = featuresets[1900:]

#posterior = prior occurences * likelihood / evidences
#naive bayes is scalable and easy to understand

#classifier = nltk.NaiveBayesClassifier.train(training_set)
##print('Accuracy percentage: ',(nltk.classify.accuracy(classifier,training_set))*100)
##classifier.show_most_informative_features(15)
#15 most informative words
#grammar is not informative
#reliability is more important than random accuracy

#save pickle is how to save your trained algorithm so training is not done again and again
#pickle saves python objects
##save_classifier = open('naivebayes.pickle','wb')
##pickle.dump(classifier,save_classifier)
#dumps the classifier in the file save_classifier
##save_classifier.close()


classifier_f = open('naivebayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
#This is used when we have to use a saved classifier
print('Accuracy percentage: ',(nltk.classify.accuracy(classifier,training_set))*100)
classifier.show_most_informative_features(15)
#this document could also be pickled
