import os
import numpy as np
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import svm
from nltk import metrics
from sklearn import metrics

stopwords = "Data/13 classify data/stopwords.txt"
train_path = "Data/13 classify data/train/"
test_data = "Data/13 classify data/test/data.txt"
test_label = "Data/13 classify data/test/label.txt"

# get Data
def getTrainingData(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in dirs:
        f =  open(os.path.join(folder_path, path), 'r', encoding="utf8") 
        for line in f:
                x = ''.join(line)
                x = gensim.utils.simple_preprocess(x) #xóa các kí tự đặc biệt
                x = ' '.join(x)
                label = ''.join(path)
                X.append(x)
                y.append(int(label[0:-4])-1)
    return (X,y)

def getTestData(file_path):
    X = []
    f = open(file_path, 'r', encoding="utf8")
    for line in f:
        x = ''.join(line)
        x = gensim.utils.simple_preprocess(x) 
        x = ' '.join(x)
        X.append(x)
    return X

def getTestLabel(file_path):
    X = []
    f = open(file_path, 'r', encoding="utf8")
    for line in f:
        x = ''.join(line)
        X.append(int(x)-1)
    return X

#######
X_train, y_train = getTrainingData(train_path)
X_test = getTestData(test_data)
y_test = getTestLabel(test_label)
# TFIDF
vectorizer = TfidfVectorizer(analyzer='word', min_df = 0.001, stop_words = getTestData(stopwords)) 
vectoX_train = vectorizer.fit_transform(X_train)
vectoX_test = vectorizer.transform(X_test)
print(vectoX_train.shape)
print(vectoX_test.shape)
#
pickle.dump(y_train, open('Trained Data/13 classify data/y_train.pkl', 'wb'))
pickle.dump(y_test, open('Trained Data/13 classify data/y_test.pkl', 'wb'))
pickle.dump(vectorizer, open('Trained Data/13 classify data/vector.pkl', 'wb'))
pickle.dump(vectoX_train, open('Trained Data/13 classify data/vectorTrain.pkl', 'wb'))
pickle.dump(vectoX_test, open('Trained Data/13 classify data/vectorTest.pkl', 'wb'))
