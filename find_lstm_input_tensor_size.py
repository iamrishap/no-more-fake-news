#!/usr/bin/python
# -*- coding:utf8 -*-

import pandas as pd
import nltk
import re
import gensim
from nltk.tokenize import word_tokenize
import numpy as np

_wnl = nltk.WordNetLemmatizer()

inputfile1 = 'test_bodies.csv'
outputfile1 = 'test_bodies.txt'
data1 = pd.read_csv(inputfile1)
data1 = data1[[u'articleBody']]

for i in range(len(data1['articleBody'])):
	data1['articleBody'][i]=_wnl.lemmatize(data1['articleBody'][i]).lower()
	data1['articleBody'][i]=" ".join(re.findall(r'\w+', data1['articleBody'][i], flags=re.UNICODE)).lower()
	data1['articleBody'][i]=data1['articleBody'][i].strip('\n')

data1['articleBody'].to_csv(outputfile1, index = False, header = False)

trainBody = pd.read_csv('train_bodies.csv') #length is 1683

# print type(trainBody['articleBody'])
# print trainBody['articleBody'].shape
#if using pre-trained word vectors, no need to join two excels
#print trainBody['articleBody'][3]
#tokenize the sentences into a list
#  of word, but not a set of words
lengthList = np.array([])
for index in range(trainBody['articleBody'].shape[0]):
    line = trainBody['articleBody'][index]
    line = line.decode('utf-8')
    #trainBody['After Splitting'][index] = str(word_tokenize(line))
    lengthList = np.append(lengthList,len(word_tokenize(line)))

print("the average length of text body is: ")
print(int(np.mean(lengthList))) #424

averageLength = np.mean(lengthList)

trainStances = pd.read_csv('train_stances.csv')
lengthList1 = np.array([])
for index in range(trainStances['Headline'].shape[0]):
    line = trainStances['Headline'][index]
    line = line.decode('utf-8')
    #trainBody['After Splitting'][index] = str(word_tokenize(line))
    lengthList1 = np.append(lengthList1,len(word_tokenize(line)))

print("the average length of headline is: ")
print(int(np.max(lengthList1)))
averageLength1 = np.max(lengthList) #45
