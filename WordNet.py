#look up definitions,synonyms,anonyms and context of words
from nltk.corpus import wordnet
syns = wordnet.synsets('program')
#synsets are a list of synonyms
#print(syns)
#here syns is a list
#print(syns[0].name())
#print(syns[0].lemmas()[0].name())
#name of lemma is plan but name of syns is plan.n.01
##print(syns[0].definition())
##
##print(syns[0].examples())

synonyms,antonyms = [],[]
for syn in  wordnet.synsets('good'):
    for l in syn.lemmas():
        #print('l:',l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
##print(set(synonyms))
##print(set(antonyms))


#symantic similarity->

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')

print(w1.wup_similarity(w2))
#0.9 similarity
#these concepts can be combined for rewriting/switching words
