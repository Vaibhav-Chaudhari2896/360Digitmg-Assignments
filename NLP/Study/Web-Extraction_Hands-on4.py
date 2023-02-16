# Extracting Data from local HTML file

# pip install bs4
from bs4 import BeautifulSoup

soup = BeautifulSoup(open('C:/Data/textmining/sample_doc.html'), 'html.parser')

soup.text

soup.contents

# Look for tag address
soup.find('address')

soup.find_all('address')


# Look for tag 'q' (this denote quotes)
soup.find_all('q')

# Look for tag 'b' (this denote texts in bold font)
soup.find_all('b')

# Look for tag 'table'
table = soup.find('table')
table

for row in table.find_all('tr'):
    columns = row.find_all('td')
    print(columns)

table.find_all('tr')[3].find_all('td')[2]


##########################################
# pip install tweepy
# Twitter Extraction
import pandas as pd
import tweepy
from tweepy import OAuthHandler
 
# Your Twittter App Credentials
# https://apps.twitter.com -> https://developer.twitter.com

consumer_key = "XmM66QbrmX0Ft8wKKDHXl4YDV"
consumer_secret = "rGNNsGdeSGlWVbREB7QUL4u9FHJK6ZpJhAaDREZhh9vxFXSx6V"
access_token = "3283097118-1MZUMi48qahfKrfZdAGS8DObfMbtAOMaCiCtSHc"
access_token_secret = "scGcNcwe6cKJhsWohroZ2gMxo4x2nGBpmoV0y1ZbYcezj"

# Calling API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# Provide the keyword you want to pull the data e.g. "Python".
keyword = "omricron"

# Fetching tweets
tweets_keyword = api.search_tweets(keyword, count=100, lang='en',
                            exclude='retweets', tweet_mode='extended') #Changed
 
for item in tweets_keyword:
    print(item)
    
tweets_for_csv = [tweet.full_text for tweet in tweets_keyword] 


# 200 tweets to be extracted 
tweets_user = api.user_timeline(screen_name="ShashiTharoor", count=200)


for item in tweets_user:
    print(item)
# Create array of tweet information: username, tweet id, date/time, text 
tweets_for_csv1 = [tweet.text for tweet in tweets_user] 

# Saving the tweets onto a CSV file
# convert 'tweets' list to pandas DataFrame
tweets_df = pd.DataFrame(tweets_for_csv1, columns=['Value'])

tweets_df.to_csv('tweets.csv')

import os
os.getcwd()
