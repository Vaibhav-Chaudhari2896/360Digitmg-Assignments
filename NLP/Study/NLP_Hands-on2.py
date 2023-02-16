# Text cleaning and Tokenization

import re

sentence5 = 'Sharat tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official \
@indian_army #India #70thRepublic_Day. For more photos ping me sharat@photoking.com :)"'

re.sub(r'([^\s\w]|_)+', ' ', sentence5).split()


# Extracting n-grams
# n-grams can be extracted from 3 different techniques:
# listed below are:
# 1. Custom defined function
# 2. NLTK
# 3. TextBlob

# Extracting n-grams using customed defined function
import re
def n_gram_extractor(input_str, n):
    tokens = re.sub(r'([^\s\w]|_)+', ' ', input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])

n_gram_extractor('The cute little boy is playing with the kitten.', 2)

n_gram_extractor('The cute little boy is playing with the kitten.', 3)


# Extracting n-grams with nltk
from nltk import ngrams
list(ngrams('The cute little boy is playing with the kitten.'.split(), 2))

list(ngrams('The cute little boy is playing with the kitten.'.split(), 3))


# Extracting n-grams using TextBlob
# TextBlob is a Python library for processing textual data.

# pip install textblob

from textblob import TextBlob
blob = TextBlob("The cute little boy is playing with the kitten.")

blob.ngrams(n=2)

blob.ngrams(n=3)


# Tokenizing texts with different packages: Keras, Textblob
sentence5 = 'Sharat tweeted, "Witnessing 70th Republic Day of India from Rajpath, New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official @indian_army #India #70thRepublic_Day. For more photos ping me sharat@photoking.com :)"'

# pip install tensorflow
# pip install keras

# Tokenization with Keras
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)

# Tokenization with TextBlob
from textblob import TextBlob
blob = TextBlob(sentence5)
blob.words

# Tokenize sentences using other nltk tokenizers:
# 1. Tweet Tokenizer
# 2. MWE Tokenizer (Multi-Word Expression)
# 3. Regexp Tokenizer
# 4. Whitespace Tokenizer
# 5. Word Punct Tokenizer


# 1. Tweet tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)

# 2. MWE Tokenizer (Multi-Word Expression)
from nltk.tokenize import MWETokenizer
mwe_tokenizer = MWETokenizer([('Republic', 'Day')]) # Declaring set of words that are to be treated as one entity
mwe_tokenizer.add_mwe(('Indian', 'Army')) # Adding more words to the set

mwe_tokenizer.tokenize(sentence5.split()) #  Indian Army' should be treated as a single token. But here "Army!" is treated as a token. 

mwe_tokenizer.tokenize(sentence5.replace('!', '').split()) # "Army!" will be treated as Army 


# 3. Regexp Tokenizer
from nltk.tokenize import RegexpTokenizer
reg_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
reg_tokenizer.tokenize(sentence5)


# 4. Whitespace Tokenizer
from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer = WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence5)


# 5. WordPunct Tokenizer
from nltk.tokenize import WordPunctTokenizer
wp_tokenizer = WordPunctTokenizer()
wp_tokenizer.tokenize(sentence5)


# Stemming
# Regexp Stemmer
sentence6 = "I love playing Cricket. Cricket players practice hard in their innings ."
from nltk.stem import RegexpStemmer
regex_stemmer = RegexpStemmer('ing$')

' '.join([regex_stemmer.stem(wd) for wd in sentence6.split()])


# Porter Stemmer
sentence7 = "Before eating, it would be nice to sanitize your hands with a sanitizer"
from nltk.stem.porter import PorterStemmer
ps_stemmer = PorterStemmer()
' '.join([ps_stemmer.stem(wd) for wd in sentence7.split()])



# Lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

sentence8 = "The codes executed today are far better than what we execute generally."

' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence8)])


# Singularize & Pluralize words
from textblob import TextBlob
sentence9 = TextBlob('She sells seashells on the seashore')
sentence9.words

sentence9.words[2].singularize()

sentence9.words[5].pluralize()


# Language Translation
# From Spanish to English

from textblob import TextBlob
en_blob = TextBlob(u'muy bien')
en_blob.translate(from_lang='es', to='en') 


# Custom Stop words removal
from nltk import word_tokenize
sentence9 = "She sells seashells on the seashore"
custom_stop_word_list = ['she', 'on', 'the', 'am', 'is', 'not']
' '.join([word for word in word_tokenize(sentence9) if word.lower() not in custom_stop_word_list])


# Extracting general features from raw texts

# Number of words
# Detect presence of wh words
# Polarity
# Subjectivity
# Language identification

import pandas as pd
df = pd.DataFrame([['The vaccine for covid-19 will be announced on 1st August.'],
                   ['Do you know how much expectation the world population is having from this research?'],
                   ['This risk of virus will end on 31st July.']])
df.columns = ['text']
df

# Number of words
from textblob import TextBlob
df['number_of_words'] = df['text'].apply(lambda x : len(TextBlob(x).words))
df['number_of_words']

# Detect presence of wh words
wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
df['is_wh_words_present'] = df['text'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False)
df['is_wh_words_present']


# Polarity
df['polarity'] = df['text'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
df['polarity']

# Subjectivity
df['subjectivity'] = df['text'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']

# Language of the sentence
df['language'] = df['text'].apply(lambda x : TextBlob(str(x)).detect_language())
df['language']


# Bag of Words
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['At least seven Indian pharma companies are working to develop a vaccine against coronavirus',
'the deadly virus that has already infected more than 14 million globally.',
'Bharat Biotech, Indian Immunologicals, are among the domestic pharma firms working on the coronavirus vaccines in India.'
]

bag_of_words_model = CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense()) # bag of words

bag_of_word_df = pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())
bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)
bag_of_word_df.head()

# Bag of word model for top 5 frequent terms
bag_of_words_model_small = CountVectorizer(max_features=5)
bag_of_word_df_small = pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
bag_of_word_df_small.columns = sorted(bag_of_words_model_small.vocabulary_)
bag_of_word_df_small.head()

# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_model = TfidfVectorizer()
print(tfidf_model.fit_transform(corpus).todense())

tfidf_df = pd.DataFrame(tfidf_model.fit_transform(corpus).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
tfidf_df.head()

# TFIDF for top 5 frequent terms
tfidf_model_small = TfidfVectorizer(max_features=5)
tfidf_df_small = pd.DataFrame(tfidf_model_small.fit_transform(corpus).todense())
tfidf_df_small.columns = sorted(tfidf_model_small.vocabulary_)
tfidf_df_small.head()


# Feature Engineering (Text Similarity)
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()

pair1 = ["Do you have Covid-19","Your body temperature will tell you"]
pair2 = ["I travelled to Malaysia.", "Where did you travel?"]
pair3 = ["He is a programmer", "Is he not a programmer?"]

def extract_text_similarity_jaccard (text1, text2):
    words_text1 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text1)]
    words_text2 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text2)]
    nr = len(set(words_text1).intersection(set(words_text2)))
    dr = len(set(words_text1).union(set(words_text2)))
    jaccard_sim = nr/dr
    return jaccard_sim

extract_text_similarity_jaccard(pair1[0], pair1[1])
extract_text_similarity_jaccard(pair2[0], pair2[1])
extract_text_similarity_jaccard(pair3[0], pair3[1])

tfidf_model = TfidfVectorizer()

# Creating a corpus which will have texts of pair1, pair2 and pair 3 respectively
corpus = [pair1[0], pair1[1], pair2[0], pair2[1], pair3[0], pair3[1]]

tfidf_results = tfidf_model.fit_transform(corpus).todense()
# Note: Here tfidf_results will have tf-idf representation of 
# texts of pair1, pair2 and pair3 in the given order.

# tfidf_results[0], tfidf_results[1] represents pair1
# tfidf_results[2], tfidf_results[3] represents pair2
# tfidf_results[4], tfidf_results[5] represents pair3

#cosine similarity between texts of pair1
cosine_similarity(tfidf_results[0], tfidf_results[1])

#cosine similarity between texts of pair2
cosine_similarity(tfidf_results[2], tfidf_results[3])

#cosine similarity between texts of pair3
cosine_similarity(tfidf_results[4], tfidf_results[5])
