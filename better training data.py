#SGD calls everything as a single class sometimes because the test data is too small wrt training data and we are using a binary classifier
#lack of data may result in volatility of accuracy
from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from statistics import mode
#we can inherit from the Classifier class


class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        #classifiers is a list of all the classifiers
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#Multinomial distribution not binary
#changing the default parameters of these algorithms can increase the success rate by about 10%
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


short_pos = open('positive.txt','r').read()
short_neg = open('negative.txt','r').read()

documents = []
for r in short_pos.split('\n'):
    documents.append( (r,'pos') )
for r in short_neg.split('\n'):
    documents.append( (r,'neg') )
all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)
#forming bag of words

for w in short_pos_words:
    all_words.append(w.lower())
#all positive words added

for w in short_neg_words:
    all_words.append(w.lower())
#all negative words added

    
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]
#top 5000 words

#other nlp concepts can also be used here

def find_features(documents):
    words = word_tokenize(documents)
    #documents was a string
    features = {}
    for w in word_features:
        features[w] = (w in words)
    #if w is in documents
    return features
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev,category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print('Original Accuracy percentage: ',(nltk.classify.accuracy(classifier,training_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MNB Accuracy percentage: ',(nltk.classify.accuracy(MNB_classifier,training_set))*100)


Bernoulli_classifier = SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print('BernoulliNB Accuracy percentage: ',(nltk.classify.accuracy(Bernoulli_classifier,training_set))*100)


LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print('Logistic Regression Accuracy percentage: ',(nltk.classify.accuracy(LR_classifier,training_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print('SGDClassifier Accuracy percentage: ',(nltk.classify.accuracy(SGDClassifier_classifier,training_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC Accuracy percentage: ',(nltk.classify.accuracy(LinearSVC_classifier,training_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC Accuracy percentage: ',(nltk.classify.accuracy(NuSVC_classifier,training_set))*100)





voted_classifier = VoteClassifier(classifier,MNB_classifier,Bernoulli_classifier,LR_classifier,SGDClassifier_classifier,LinearSVC_classifier,NuSVC_classifier)
print('Voted_Classifer Accuracy percentage: ',(nltk.classify.accuracy(voted_classifier,training_set))*100)

print('classification', voted_classifier.classify(testing[0][0]), 'confidence %',voted_classifier.confidence(testing[0][0]))
print('classification', voted_classifier.classify(testing[1][0]), 'confidence %',voted_classifier.confidence(testing[1][0]))
#pickle all classifiers, documents, all_words, featuresets and word_features to save time 
