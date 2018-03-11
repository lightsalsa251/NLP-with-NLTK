from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
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

SVC_classifier = SklearnClassifier(SVC())
#Skleaern classifier is a wrapper around the nltk classifier
SVC_classifier.train(training_set)
print('SVC Accuracy percentage: ',(nltk.classify.accuracy(SVC_classifier,training_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
#Skleaern classifier is a wrapper around the nltk classifier
LinearSVC_classifier.train(training_set)
print('LinearSVC Accuracy percentage: ',(nltk.classify.accuracy(LinearSVC_classifier,training_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
#Skleaern classifier is a wrapper around the nltk classifier
NuSVC_classifier.train(training_set)
print('NuSVC Accuracy percentage: ',(nltk.classify.accuracy(NuSVC_classifier,training_set))*100)

#customise these algorithms to increase accuracy
#confidency score can also be computed
