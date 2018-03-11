from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from nltk.classify import ClassifierI
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



classifier_f = open('naivebayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
print('Original Accuracy percentage: ',(nltk.classify.accuracy(classifier,training_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
#Skleaern classifier is a wrapper around the nltk classifier
MNB_classifier.train(training_set)
print('MNB Accuracy percentage: ',(nltk.classify.accuracy(MNB_classifier,training_set))*100)

##Gaussian_classifier = SklearnClassifier(GaussianNB())
###Skleaern classifier is a wrapper around the nltk classifier
##Gaussian_classifier.train(training_set)
##print('GNB Accuracy percentage: ',(nltk.classify.accuracy(Gaussian_classifier,training_set))*100)

Bernoulli_classifier = SklearnClassifier(BernoulliNB())
#Skleaern classifier is a wrapper around the nltk classifier
Bernoulli_classifier.train(training_set)
print('BernoulliNB Accuracy percentage: ',(nltk.classify.accuracy(Bernoulli_classifier,training_set))*100)

#Voting system can be used for finding the output if we are using several algorithms

LR_classifier = SklearnClassifier(LogisticRegression())
#Skleaern classifier is a wrapper around the nltk classifier
LR_classifier.train(training_set)
print('Logistic Regression Accuracy percentage: ',(nltk.classify.accuracy(LR_classifier,training_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
#Skleaern classifier is a wrapper around the nltk classifier
SGDClassifier_classifier.train(training_set)
print('SGDClassifier Accuracy percentage: ',(nltk.classify.accuracy(SGDClassifier_classifier,training_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
#Skleaern classifier is a wrapper around the nltk classifier
LinearSVC_classifier.train(training_set)
print('LinearSVC Accuracy percentage: ',(nltk.classify.accuracy(LinearSVC_classifier,training_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
#Skleaern classifier is a wrapper around the nltk classifier
NuSVC_classifier.train(training_set)
print('NuSVC Accuracy percentage: ',(nltk.classify.accuracy(NuSVC_classifier,training_set))*100)


#New classifier will have voting system of all the algorithms used
#it raises accuracy, reliavbility
#confidence parameter will also be calculated
#svc gave very low accuracy so it is removed from voting



voted_classifier = VoteClassifier(classifier,MNB_classifier,Bernoulli_classifier,LR_classifier,SGDClassifier_classifier,LinearSVC_classifier,NuSVC_classifier)
print('Voted_Classifer Accuracy percentage: ',(nltk.classify.accuracy(voted_classifier,training_set))*100)

print('classification', voted_classifier.classify(testing[0][0]), 'confidence %',voted_classifier.confidence(testing[0][0]))
print('classification', voted_classifier.classify(testing[1][0]), 'confidence %',voted_classifier.confidence(testing[1][0]))
#voting is done for every feature i.e. sentence, here only one two sentence is used  
#we only need to update the test_set list for a real time project
#pass any dataset to find features and then pickle it
