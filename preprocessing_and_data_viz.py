
# coding: utf-8

#Import all required modules
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack



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
                 train_bodies_df[['Body_ID', 'articleBody']])

test_df = pd.merge(test_df_tmp,
                 test_bodies_df[['Body_ID', 'articleBody']])

train_df = train_df.rename(columns={'articleBody': 'Body_Text'})
test_df = test_df.rename(columns={'articleBody': 'Body_Text'})

train_df = train_df[['Headline','Body_Text','Stance']]
train_df.groupby('Stance').describe()

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

train_df['length'] = train_df['Body_Text'].apply(len)
train_df['length'].plot.hist(bins=50)
train_df.hist(column='length',by='Stance')

