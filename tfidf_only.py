
# coding: utf-8

# ## TF-IDF Feature Extraction and Random Forest Classifier
#
# The following sources were used to construct this Jupyter Notebook:
#
# * [Numpy: Dot Multiplication, Vstack, Hstack, Flatten](https://www.youtube.com/watch?v=nkO6bmp511M)
# * [Scikit Learn TF-IDF Feature Extraction and Latent Semantic Analysis](https://www.youtube.com/watch?v=BJ0MnawUpaU)
# * [Fake News Challenge TF-IDF Baseline](https://github.com/gmyrianthous/fakenewschallenge/blob/master/baseline.py)
# * [Python TF-IDF Algorithm Built From Scratch](https://www.youtube.com/watch?v=hXNbFNCgPfY)
# * [Theory Behind TF-IDF](https://www.youtube.com/watch?v=4vT4fzjkGCQ)

#Import all required modules
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import score


#Import data from CSV file and create a dataframe
def create_dataframe(filename):
    #Read file into a pandas dataframe
    df = pd.read_csv(filename)
    #Remove white space in column names
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df


#Create dataframes for both training and testing sets
train_df_tmp = create_dataframe('train_stances.csv')
test_df_tmp = create_dataframe('competition_test_stances.csv')
train_bodies_df = create_dataframe('train_bodies.csv')
test_bodies_df = create_dataframe('test_bodies.csv')

train_df_tmp.head(5)

train_df = pd.merge(train_df_tmp,
                 train_bodies_df[['Body_ID', 'articleBody']],
                 on='Body_ID')

test_df = pd.merge(test_df_tmp,
                 test_bodies_df[['Body_ID', 'articleBody']],
                 on='Body_ID')

train_df = train_df.rename(columns={'articleBody': 'Body_Text'})
test_df = test_df.rename(columns={'articleBody': 'Body_Text'})
# test_df.sort_values(by=['Body_ID']).head(5)
# train_df.sort_values(by=['Body_ID']).head(5)

#Apply Scikit Learn TFIDF Feature Extraction Algorithm
body_text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english',max_features=1024)
headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english',max_features=1024)

#Create vocabulary based on training data
train_body_tfidf = body_text_vectorizer.fit_transform(train_df['Body_Text'])
train_headline_tfidf = headline_vectorizer.fit_transform(train_df['Headline'])

#Use vocabulary for testing data
test_body_tfidf = body_text_vectorizer.transform(test_df['Body_Text'])
test_headline_tfidf = headline_vectorizer.transform(test_df['Headline'])

#Tuple represents (Instance, Feature); value to the right of the tuple
#represents the feature's tf-idf score
# print(train_body_tfidf)

#Merge headline and body_text tf-idf vectors together - Stack arrays in sequence horizontally
train_features = hstack([train_body_tfidf, train_headline_tfidf])
test_features = hstack([test_body_tfidf, test_headline_tfidf])


#Feature vector (the headline and body text features merged by hstack)
# print(train_features)

#Extract training and test labels
train_labels = list(train_df['Stance'])
test_labels = list(test_df['Stance'])

#Initialize random forest classifier (Scikit Learn)
rf_classifier = RandomForestClassifier(n_estimators=10)
y_pred = rf_classifier.fit(train_features, train_labels).predict(test_features)
accuracy_score(test_labels, y_pred)

#Initialize multinomialnb classifier
nb_classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
y_pred = nb_classifier.fit(train_features,train_labels).predict(test_features)

accuracy_score(test_labels, y_pred)
score.report_score(test_labels,y_pred)

#Add predicted labels to test dataframe
#test_df['RF_Predicted_Stance'] = list(y_pred)
#test_df2 = test_df[['Headline','Body_Text','RF_Predicted_Stance','Stance']]
#test_df2[test_df2['RF_Predicted_Stance'] == 'unrelated']

