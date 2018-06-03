# coding: utf-8

# ###### Cosine Feature Extraction and Random Forest Classifier
# 
# The following sources were used to construct this Jupyter Notebook:
# 
# * [Numpy: Dot Multiplication, Vstack, Hstack, Flatten](https://www.youtube.com/watch?v=nkO6bmp511M)
# * [Scikit Learn TF-IDF Feature Extraction and Latent Semantic Analysis](https://www.youtube.com/watch?v=BJ0MnawUpaU)
# * [Fake News Challenge TF-IDF Baseline](https://github.com/gmyrianthous/fakenewschallenge/blob/master/baseline.py)
# * [Python TF-IDF Algorithm Built From Scratch](https://www.youtube.com/watch?v=hXNbFNCgPfY)
# * [Theory Behind TF-IDF](https://www.youtube.com/watch?v=4vT4fzjkGCQ)
# * [Plotting Classifier Boundaries](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)


# Import all required modules

# For parsing and visualizing data
from pandas import DataFrame, read_csv
import pandas as pd

# For visualizing data
import matplotlib.pyplot as plt

# For processing data
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import baseline_features
from sklearn.decomposition import TruncatedSVD

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# For scoring
from sklearn.metrics import accuracy_score
import score  # Score used in competition

# Progress Bar
from tqdm import tqdm


# Reloading modules that have been updated
# import importlib
# importlib.reload(baseline_features)


# # Data Preparation

# ## Create Dataframes

# Import data from CSV file and create a dataframe
def create_dataframe(filename):
    # Read file into a pandas dataframe
    df = pd.read_csv(filename)
    # Remove white space in column names
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df


# Create dataframes for both training and testing sets
train_df_tmp = create_dataframe('train_stances.csv')
train_bodies_df = create_dataframe('train_bodies.csv')

test_df_tmp = create_dataframe('competition_test_stances.csv')
test_bodies_df = create_dataframe('test_bodies.csv')

train_df_tmp.head(5)

# ## Join Dataframes on Body_ID


train_df = pd.merge(train_df_tmp,
                    train_bodies_df[['Body_ID', 'articleBody']],
                    on='Body_ID')

test_df = pd.merge(test_df_tmp,
                   test_bodies_df[['Body_ID', 'articleBody']],
                   on='Body_ID')

train_df = train_df.rename(columns={'articleBody': 'Body_Text'})
test_df = test_df.rename(columns={'articleBody': 'Body_Text'})

# Split training data into training and validation set
train_df, validate_df, train_labels, validate_labels = train_test_split(train_df[['Body_Text', 'Headline']],
                                                                        train_df['Stance'], test_size=.4,
                                                                        random_state=42)

# # Feature Engineering

# ## TF-IDF Features

# Apply Scikit Learn TFIDF Feature Extraction Algorithm
body_text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', max_features=1024)
headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', max_features=1024)

# Create vocabulary based on training data
train_body_tfidf = body_text_vectorizer.fit_transform(train_df['Body_Text'])
train_headline_tfidf = headline_vectorizer.fit_transform(train_df['Headline'])

# Create vocabulary based on validation data
validate_body_tfidf = body_text_vectorizer.transform(validate_df['Body_Text'])
validate_headline_tfidf = headline_vectorizer.transform(validate_df['Headline'])

# Use vocabulary for testing data
test_body_tfidf = body_text_vectorizer.transform(test_df['Body_Text'])
test_headline_tfidf = headline_vectorizer.transform(test_df['Headline'])


# ## Cosine Similarity Features

# Cosine Similarity
def get_cosine_similarity(body_tfidf, headline_tfidf):
    cosine_features = []
    # len body_tfidf = len headline_tfidf
    for i in tqdm(range(body_tfidf.shape[0])):
        cosine_features.append(
            cosine_similarity((body_tfidf.A[0].reshape(1, -1)), (headline_tfidf.A[0].reshape(1, -1)))[0][0])
    return np.array(cosine_features).reshape(body_tfidf.shape[0], 1)

# Leave this commented out unless you are re-calculating the cosine similarity
# which can be found in the pickle files labeled:
# train_cosine_features.p and test_cosine_features.p

# Train data
train_cosine_features = get_cosine_similarity(train_body_tfidf,train_headline_tfidf)

# Validate data
validate_cosine_features = get_cosine_similarity(validate_body_tfidf,validate_headline_tfidf)

# Test data
test_cosine_features = get_cosine_similarity(test_body_tfidf,test_headline_tfidf)

pickle.dump(train_cosine_features,open('train_cosine_features.p','wb'))
pickle.dump(validate_cosine_features,open('validate_cosine_features.p','wb'))
pickle.dump(test_cosine_features,open('test_cosine_features.p','wb'))

train_cosine_features = pickle.load(open('train_cosine_features.p', 'rb'))
validate_cosine_features = pickle.load(open('validate_cosine_features.p', 'rb'))
test_cosine_features = pickle.load(open('test_cosine_features.p', 'rb'))

# ## Hand Selected Features (Baseline Features)

train_hand_features = baseline_features.hand_features(train_df['Headline'], train_df['Body_Text'])
validate_hand_features = baseline_features.hand_features(validate_df['Headline'], validate_df['Body_Text'])
test_hand_features = baseline_features.hand_features(test_df['Headline'], test_df['Body_Text'])


train_hand_features = np.array(train_hand_features)
validate_hand_features = np.array(validate_hand_features)
test_hand_features = np.array(test_hand_features)

# ## Word Overlap Features (Baseline Feature)

train_overlap_features = baseline_features.word_overlap_features(train_df['Headline'], train_df['Body_Text'])
validate_overlap_features = baseline_features.word_overlap_features(validate_df['Headline'], validate_df['Body_Text'])
test_overlap_features = baseline_features.word_overlap_features(test_df['Headline'], test_df['Body_Text'])

train_overlap_features = np.array(train_overlap_features)
validate_overlap_features = np.array(validate_overlap_features)
test_overlap_features = np.array(test_overlap_features)

# ## Polarity Features (Baseline Feature)

train_polarity_features = baseline_features.polarity_features(train_df['Headline'], train_df['Body_Text'])
validate_polarity_features = baseline_features.polarity_features(validate_df['Headline'], validate_df['Body_Text'])
test_polarity_features = baseline_features.polarity_features(test_df['Headline'], test_df['Body_Text'])

train_polarity_features = np.array(train_polarity_features)
validate_polarity_features = np.array(validate_polarity_features)
test_polarity_features = np.array(test_polarity_features)

# ## Refuting Features (Baseline)

train_refuting_features = baseline_features.refuting_features(train_df['Headline'], train_df['Body_Text'])
validate_refuting_features = baseline_features.refuting_features(validate_df['Headline'], validate_df['Body_Text'])
test_refuting_features = baseline_features.refuting_features(test_df['Headline'], test_df['Body_Text'])

train_refuting_features = np.array(train_refuting_features)
validate_refuting_features = np.array(validate_refuting_features)
test_refuting_features = np.array(test_refuting_features)

# ## Concatenate feature vectors

train_features = hstack([
    train_body_tfidf,
    train_headline_tfidf,
    train_hand_features,
    train_cosine_features,
    train_overlap_features,
    train_polarity_features,
    train_refuting_features

])
validate_features = hstack([
    validate_body_tfidf,
    validate_headline_tfidf,
    validate_hand_features,
    validate_cosine_features,
    validate_overlap_features,
    validate_polarity_features,
    validate_refuting_features

])
test_features = hstack([
    test_body_tfidf,
    test_headline_tfidf,
    test_hand_features,
    test_cosine_features,
    test_overlap_features,
    test_polarity_features,
    test_refuting_features
])

# # Classification

# ## Extract labels
# We already have train_labels and validate_labels from before
test_labels = list(test_df['Stance'])

# ## Run Classifiers and Score Validation Output

names = ["Random Forest", "Multinomial Naive Bayes", "Gradient Boosting", "K Nearest Neighbors", "Linear SVM", "Decision Tree",
         "Logistic Regression"]

# "K Nearest Neighbors",
classifiers = [
    RandomForestClassifier(n_estimators=10),
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
    GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True),
    KNeighborsClassifier(4),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    LogisticRegression(C=1e5)
]

# KNeighborsClassifier(4, algorithm='kd_tree'),
for n, clf in zip(names, classifiers):
    print(n)
    y_pred = clf.fit(train_features, train_labels).predict(validate_features)
    print(score.report_score(test_labels, y_pred))
    print('\n')

# ## Run Classifiers and Score Test Output

# This is how well we would have scored in the actual competition
names = ["Random Forest", "Multinomial Naive Bayes", "Gradient Boosting", "K Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Logistic Regression"]

classifiers = [
    RandomForestClassifier(n_estimators=10),
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
    GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True),
    KNeighborsClassifier(4),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    LogisticRegression(C=1e5)
]

for n, clf in zip(names, classifiers):
    print(n)
    y_pred = clf.fit(train_features, train_labels).predict(test_features)
    print(score.report_score(test_labels, y_pred))
    print('\n')
