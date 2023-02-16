import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Loading the data set
email_data = pd.read_csv("C:/Users/vaibh/Desktop/360 Digitmg/Naive Bayes/sms_raw_NB.csv", encoding = "ISO-8859-1")
email_data
# cleaning data 

# ---------
stop_words = []
# Load the custom built Stopwords
with open("C:/Users/vaibh/Desktop/360 Digitmg/NLP/Study/stop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
# ---------

import re

def cleaning_text(data):
    i = re.sub("[^A-Za-z" "]+", " ", data).lower()
#    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word) > 3:
            w.append(word)
    return (" ".join(w))

# testing above function with sample text => removes punctuations, numbers
cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi hw .. are you?")
cleaning_text("you know what would be amazingly awesome? BOS-FLL PLEASE!!!!!!! I want to fly with only you.")

email_data.text = email_data.text.apply(cleaning_text)

# removing empty rows
email_data = email_data.loc[email_data.text != "", :]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

email_train, email_test = train_test_split(email_data, test_size = 0.2)


# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
emails_bow = CountVectorizer(analyzer = split_into_words).fit(email_data.text)

# Defining BOW for all messages
all_emails_matrix = emails_bow.transform(email_data.text)

# For training messages
train_emails_matrix = emails_bow.transform(email_train.text)

# For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, email_train.type)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)

pd.crosstab(test_pred_m, email_test.type)

accuracy_test_m = np.mean(test_pred_m == email_test.type)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, email_test.type) 


# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == email_train.type)
accuracy_train_m

pd.crosstab(train_pred_m, email_train.type)

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

#formula: 
# P(w|spam) = (num of spam with w + alpha)/(Total num of spam emails + K(alpha))
# K = total num of words in the email to be classified

classifier_mb_lap = MB(alpha = 0.75)
classifier_mb_lap.fit(train_tfidf, email_train.type)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == email_test.type)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, email_test.type) 

pd.crosstab(test_pred_lap, email_test.type)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == email_train.type)
accuracy_train_lap

pd.crosstab(train_pred_lap, email_train.type)
