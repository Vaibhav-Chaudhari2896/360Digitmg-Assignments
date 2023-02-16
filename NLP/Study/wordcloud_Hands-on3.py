# WordCloud

# pip install WordCloud

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

with open("C:\\Datasets_BA\\360DigiTMG\\DS_India\\360DigiTMG DS India Module wise PPTs\\Module 18 Text Mining_Natural Language Processing (NLP)\\Data\\oneplusseven.txt","r",encoding="utf-8") as sw:
    oneplus = sw.read()
    
other_stopwords_to_remove = ['\\n', 'n', '\\', '>', 'nLines', 'nI',"n'", "hi"]
STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))
stopwords = set(STOPWORDS)
text = str(oneplus)
wordcloud = WordCloud(width = 1800, height = 1800, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Dependency Parse Tree and Named Entities - Visualizations
# Install the required libraries from Python console

# pip install spacy
# python -m spacy download en

# The visualizations will work in jupyter 
################
# import spacy
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp('God helps those who help themselves.')
displacy.serve(doc, style='dep')

displacy.render(doc, style='dep')

text = 'Once upon a time there lived a programmer named Sharat Chandra. \
        He along with his close friend Bharani Kumar who was also known as Star \
        who is the founder of 360DigiTMG and AiSPRY, trained people on NLP.'
doc2 = nlp(text)
displacy.render(doc2, style='ent', jupyter=True)
################


# Building pipelines for NLP projects
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn import tree
# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# email_data = pd.read_csv("C:/Datasets_BA/Data Mining-Supervised/sms_raw_NB.csv", encoding = "ISO-8859-1")
# data = email_data.text
# data_categories = ['ham', 'spam']

# categories = ['misc.forsale', 'sci.electronics', 'talk.religion.misc']
# news_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, download_if_missing=True)

# text_classifier_pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
# text_classifier_pipeline.fit(news_data.data, news_data.target) 

# res = pd.DataFrame(text_classifier_pipeline.fit_transform(news_data.data, news_data.target).todense()).head()


# Saving and Loading models

# pip install joblib
import pickle
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
'Data Science is the most in demand job role in the current market',
'It is combination of both Maths and Business skills at a time',
'Natural Language Processing is a part of Data Science'
]
tfidf_model = TfidfVectorizer()
print(tfidf_model.fit_transform(corpus).todense())

dump(tfidf_model, 'tfidf_model.joblib')

import os
os.getcwd()

# new corpus
text = ['Once upon a time there lived a programmer named Sharat Chandra. \
        He along with his close friend Bharani Kumar who was also known as Star \
        who is the founder of 360DigiTMG and AiSPRY, trained people on NLP.']

tfidf_model_loaded = load('tfidf_model.joblib')

print(tfidf_model_loaded.fit_transform(text).todense())


# Pickle library
pickle.dump(tfidf_model, open("tfidf_model.pickle.dat", "wb")) #Save the model
loaded_model = pickle.load(open("tfidf_model.pickle.dat", "rb")) #To load the saved model

print(loaded_model.fit_transform(text).todense())
