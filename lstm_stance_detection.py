#!/usr/bin/python
# -*- coding:utf8 -*-
from keras.layers import Dense,LSTM
from keras.utils import to_categorical
from keras.models import Sequential
from keras import optimizers
import keras
import pandas as pd
import gensim
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#the designated length for the size of headlines and body text in the input tensor
lengthForBody = 424
lengthForHeadline = 45

indice = 2000
import numpy as np
import pandas as pd
import os
from nltk.tokenize import word_tokenize
import gensim
print("start loading w2v")
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
print("loading completed")

#using GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_DEVICE_DEVICES"]= "0"

#join two tables
#train_df_tmp = pd.read_csv('train_stances.csv')
#train_bodies_df = pd.read_csv('train_bodies.csv')
#train_df = pd.merge(train_df_tmp,train_bodies_df[['Body ID', 'articleBody']],on='Body ID')

#train_df.to_csv('train_bodies_joined.csv')

# file *joined.csv is the *stance.csv joined with *bodies.csv
dataset = pd.read_csv('train_bodies_joined.csv')

#three vectors: two vectors for features and one vector for label
headlines, bodies, stances = dataset['Headline'], dataset['articleBody'], dataset['Stance']
from sklearn import preprocessing

#make the string label into discrete numeric
stances = preprocessing.LabelEncoder().fit_transform(stances)

for index in range(indice): #len(headlines)
    line = headlines[index]
    line = line.decode('utf-8')
    headlines[index] = word_tokenize(line)

#try and except block: the tokenized word may be exist at word2Vec dictionary, an error may occur
headlineList = []
for eachSentence in headlines[0:indice]:
    headlineLi = []
    print eachSentence
    for eachword in eachSentence:
        try:
            headlineLi.append(model[eachword])
            #the type of word vectors is <type 'numpy.ndarray'>
        except:
            pass
    headlineList.append(headlineLi)

headlineList = np.array(headlineList)

#padding the headline, since the length of each headline is not equal
headlineList = pad_sequences(headlineList,padding='post',maxlen=lengthForHeadline,value=0.0,dtype='float32')

print "for headlines input"
print headlineList.shape
#for each in headlineList:
    #print len(each)
    #print each

bodyTextList = []
for eachSentence in bodies[0:indice]:
    bodyTextLi = []
    #print eachSentence
    for eachword in eachSentence:
        try:
            bodyTextLi.append(model[eachword])
            #the type of word vectors is <type 'numpy.ndarray'>
        except:
            pass
    bodyTextList.append(bodyTextLi)

bodyTextList = np.array(bodyTextList)

#padding the body text, since the length of each body text is not equal
bodyTextList = pad_sequences(bodyTextList,padding='post',maxlen=lengthForBody,value=0.0,dtype='float32')

print "for bodyText input:"
print bodyTextList.shape
#for each in bodyTextList:
    #print len(each)
    #print each
data_x = np.append(headlineList,bodyTextList,axis=1)
data_y = stances[0:indice]
#now we get the correct tensor format as input for LSTM

#randomly shuffle the examples
data_x,data_y = shuffle(data_x,data_y)

#split the samples into 8:2 training and testing set
# !!!Note: we will still be using the actual test set the competition provides us to evaluate our model
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train,num_classes=4)
y_test = to_categorical(y_test,num_classes=4)


#set the size for input tensor
nb_lstm_output = 4
nb_time_steps = lengthForBody + lengthForHeadline #number of column
nb_input_vector = 300

#build model
model = Sequential()
model.add(LSTM(units=nb_lstm_output,input_shape=(nb_time_steps,nb_input_vector)))
model.add(Dense(4,activation='softmax',name = 'dense1'))


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd ,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=100,batch_size=5000,verbose=1)

score=model.evaluate(x_test,y_test,batch_size=1000,verbose=1)

# serialize model to JSON and next time for prediction or evaluate our model with the actual "unlabelled" testing set
# we don't have to run the LSTM again, just load the model and the weights usingh *.json and *.h5
model_json = model.to_json()
with open("LSTM_for_stance_detection.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("LSTM_weight_for_stance_detection.h5")
print("Saved model to disk")

print(score)