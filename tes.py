from __future__ import print_function, division
import nltk
import os
import random
from nltk import word_tokenize, WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem.lancaster import *
from nltk.corpus import stopwords
from collections import Counter
from nltk import NaiveBayesClassifier, classify ,SklearnClassifier ,MaxentClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix 
from sklearn.feature_extraction.text import CountVectorizer

#Stoplist di bahasa inggris
#Stoplist itu berisi kata-kata yang gk penting yang bakal di remove contoho (so, only, is ,a , etc)
stoplist = stopwords.words('english')

#Preprocess data 
# 1.Tokenization buat kata jadi token ilangin tanda baca atau simbol"
# 2.Stemming / Lemmatization buat kata jadi kata dasar (katanya yang lebih bagus lemmatize)
# 3.Ubah semua kata jadi lowercase supaya kata yang sama dianggap satu fitur
def preprocess(sentence):
	tokens = word_tokenize(sentence)
	lemmatizer = WordNetLemmatizer() #Ubah jadi kata dasar , bisa ganti kata contoh (better jadi good)
	stemmerizer  = PorterStemmer() #Motong kata jadi kata dasar, bisa ganti kata Leaves jadi Leav,  bagusnya lebih cepet
	#stemmerizer = LancasterStemmer()
	#return [lemmatizer.lemmatize(word.lower()) for word in tokens]
	return [stemmerizer.stem(word.lower()) for word in tokens]

#Extract features
#1 Baca email
#2 Preprocess email text
#3 Kalau pake BOW(Bag of word) :
#	Hitung frekuensi kemunculan katanya
#4 Kalau gk pake BOW:
#   Cukup di data katanya gk usah diitung
def get_features(text, is_bow):
    if  is_bow.lower() == 'y':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

#Buat list isi text email
def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r', encoding = "ISO-8859-1")
        a_list.append(f.read())
    f.close()
    return a_list

#Fungsi buat train data 
#Katanya paling bagus Naive Bayes buat spam filter , tapi untuk variannya belom tau
#SVC bisa lebih bagus dari Naives Bayes kalau featurenya bnyk 
#Buat Max entropy bnyk algoritma buat eksplorasi ama parameter iterasi
def train(features, samples_proportion,classifier_choose):
    train_size = int(len(features) * samples_proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    classifier = NaiveBayesClassifier.train(train_set)
    if classifier_choose == 1:
    	classifier = NaiveBayesClassifier.train(train_set)
    elif classifier_choose == 2:
    	classifier = SklearnClassifier(MultinomialNB()).train(train_set)
    elif classifier_choose == 3:
    	classifier = SklearnClassifier(GaussianNB()).train(train_set)
    elif classifier_choose == 4:
    	classifier = SklearnClassifier(BernoulliNB()).train(train_set)
    elif classifier_choose == 5:
    	classifier = SklearnClassifier(SVC(), sparse=False).train(train_set)
    elif classifier_choose == 6:
    	#Bisa pilih algorithm ama masukin parameter ketigas buat tentuin brapa kali iterasi
    	#Makin banyak iterasi makin accurassi makin bagus (mungkin masih gk yakin)
        classifier = MaxentClassifier.train(train_set, MaxentClassifier.ALGORITHMS[0])

    return train_set, test_set, classifier

#Fungsi buat keluarin hasil evaluasi
def evaluate(train_set, test_set, classifier):
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
 

   
if __name__ == "__main__":
	#ASUMSI  = DATA EMAIL BUKAN DATA MENTAH YANG MENGANDUNG HTML TAG, TAPI UDAH BERUPA TEKS
	#folder isi data email text yang  spam bisa ganti 6 folder
	spam = init_lists('enron1/spam/') 
	#folder isi data email text yang  ham bisa ganti 6 folder
	ham = init_lists('enron1/ham/') 
	all_emails = [(email, 'spam') for email in spam]
	all_emails += [(email, 'ham') for email in ham]
	random.shuffle(all_emails)
	print ('Data size = ' + str(len(all_emails)) + ' emails')
	is_bow = input('Using bow ? (Y/N) ')
	all_features = [(get_features(email, is_bow), label) for (email, label) in all_emails]
	print ('Collected ' + str(len(all_features)) + ' feature sets')
	samples_proportion = float(input('Distributed number for split data train-test: '))
	classifier_choose = int(input('Choose your classifier: '))
	train_set, test_set, classifier = train(all_features, samples_proportion,classifier_choose)
	evaluate(train_set, test_set, classifier)

