###  Fake News Challenge - Brogrammers  ###

Greetings from Team "Brogrammers". This is the project developed as a solution to Fake News Challenge (http://www.fakenewschallenge.org) by the team comprising of John Fantell, Yibo Chai, and Rishap Sharma.


###  Summary  ###

While this project was not submitted as part of the competition, all of the results are evaluated using the same scoring methodology used in the competition. Ten different features were used in total: TF-IDF, Cosine Similarity, Word2Vec, Doc2Vec, Word Overlap, Word Polarity, Refuting Words, Binary Cooccurrences, Binary Cooccurrence Stops, and Count grams (Figure 1). The latter five features were baseline features provided by the competition website. For predicting class labels various classifiers were used namely random forest, multinomial naive bayes, gradient boosting, k nearest neighbours, linear svm, decision tree, logistic regression, and LSTM.

###  Result  ###

Using feature engineering, various ML classifier and also ensemble learning models, applying multi-stage classification strategy, and finally with a deep LSTM model, we got a optimal solution of score of 81.999% for this stance detection competition.


###  Requirements   ###

python>=3.4.0


###  Installation   ###

python3.4 -m pip install -r requirements.txt --upgrade


###  Dependencies   ###

In order to run this project, please download the dataset (features and model) from this Google Drive folder (https://drive.google.com/open?id=1uyZd5HaZAG6dW8-Hg_5ysC9Ji2FyHaDV) and place in the root folder. Please refer GitHub repository (https://github.com/jfantell/ML_Assignment2) for the latest updates.

Note: As the word2vector requires the file to be input in the txt format, we use the file csv_to_txt.py to convert the input file from csv to txt format. As the LSTM model requires that the input is not sparse, we find the designated length of the Headline and Body text to be used by tensor. Average length for Body text is 424 and for the Headline it is 45.
