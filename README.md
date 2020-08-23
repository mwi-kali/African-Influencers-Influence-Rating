# African influencers: Twitter users segmentation
## Introduction

The information age brought change to how businesses are conducted. This gave rise to e-marketing. In the same age,social media applications were developed and created an online platform for businesses to reach their customers on a social basis. However, businesses noticed that people on social media do not like the interruption brought by advertisements from brands. This created a new space for people to advertise the businesses through their social media platform to make it less of an advert hence more receptive to the audience.This created a new e-marketing term called influencer marketing.

Social media platforms are quickly evolving and with that getting more and more users.The development of countries in Africa is a factor to the robust growth of the numbers of social media users. Twitter users are 5.66% of all Social Media users in Africa (StatCounterGlobal Stats, 2020). This ranks it fourth after Facebook, YouTube and Pinterest.

Most companies approach social media influencers based on the number of followers they have. 

The aim of this activity is to identify the influencers with the most influence and rank them accordingly. 

### Prerequisites

Installed Anaconda. (https://www.anaconda.com/products/individual)

A Twitter developer account( https://developer.twitter.com/en/apps). 

## Let's Get Started

As a starting point, known influential usernames of user accounts and government related accounts that are common in the Africa space have been used.
The Twitter data of these chosen usernames has been collected and cleaned before processing.
The influencers' influence will be measured through the number of people they reach, their popularity and even their relevance.


### Step 1. Data Collection

#### Part A. List of African leaders and influential twitter users

The data was obtained from two websites. The websites were scraped to get the list of African leaders on twitter and influential twitter users. 

Data sources Websites used for web scraping:

1. 100 most influential Twitter users in Africa by Africa Freak Link - https://africafreak.com/100-most-influential-twitter-users-in-africa. 

2. African leaders respond to coronavirus… on Twitter AfricaSource by Luke Tyburski Link - https://www.atlanticcouncil.org/blogs/africasource/african-leaders-respond#-to-corona virus-on-twitter/#east-africa

```
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import pandas as pd
import os, sys
import requests
import urllib.request
import time
import fire
import re
# Extracting data from website for 100 most influential Twitter users in Africa
page_url = 'https://africafreak.com/100-most-influential-twitter-users-in-africa'
page = requests.get(page_url)
soup = BeautifulSoup(page.content, 'html.parser')
data = soup.find(id='content-area')
#Filtering to find the name and username of the 100 most influential Twitter users in Africa
influencer_name = data.find_all('h2')
africa_influencers = []
for influencer in influencer_name:
    influencer = influencer.text
    africa_influencers.extend([influencer])
    
africa_influencers = [word.replace(".", "").replace("(", "").replace(")", "") for word in africa_influencers]
africa_influencers = africa_influencers[:-4]
#into a dataframe
africa_influencers = [influencer.split('@') for influencer in africa_influencers]
africa_influencers_data = pd.DataFrame(africa_influencers)
#Small Clean up
def remove(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list] 
    return list
#using the clean up
africa_influencers_data[0] = remove(africa_influencers_data[0])
#rename columns
africa_influencers_data = africa_influencers_data.rename(columns={0:'Name',1:'Username'})
#remove empty column
africa_influencers_data = africa_influencers_data.drop(columns = 2)
# Extracting data from website for African leaders respond to coronavirus… on Twitter another_page_url = 'https://www.atlanticcouncil.org/blogs/africasource/african-leaders-respond-to-coronavirus-on-twitter/#east-africa' another_page = requests.get(another_page_url) another_soup = BeautifulSoup(another_page.content, 'html.parser') more_data = another_soup.find(id="content")
#Filtering to find the name and username for African leaders who responded to coronavirus… on Twitter
leader_name = more_data.find_all('blockquote')
splitting = []
for leader in leader_name:
    leader = leader.text
    leader = leader.split('-')
    splitting.extend([leader[-1]])
african_leaders = []
for leader in splitting:
    leader = leader.split('March')
    african_leaders.extend([leader[0]])
#Small Clean up
def special_remove(list): 
    list = [word.replace("(", "").replace(")", "")for word in list]
    return list
#using cleanup
african_leaders = special_remove(african_leaders)
#to dataframe
african_leaders = [leader.split('@') for leader in african_leaders]
african_leaders_data = pd.DataFrame(african_leaders)
african_leaders_data = african_leaders_data.rename(columns={0:'Name',1:'Username'})
leader_usernames = list(african_leaders_data['Username'])
influencer_usernames = list(africa_influencers_data['Username'])
leader_usernames = [username.replace(" ", "") for username in leader_usernames]
pd.DataFrame(leader_usernames).to_csv('data/leader_usernames.csv')
pd.DataFrame(influencer_usernames).to_csv('data/influencer_usernames.csv')
```

#### Part B. Twitter Data

From the list, the Twitter API was accessed to get data on the African leaders on twitter and influential twitter users, to get their tweets and to get tweets that mention the the list of African leaders on twitter and influential twitter users.

```
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from collections import Counter
from datetime import datetime, date, time, timedelta
import twitter
#Import the necessary methods from tweepy library
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor

#sentiment analysis package
from textblob import TextBlob

#general text pre-processor
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

#tweet pre-processor 
import preprocessor as p

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path  # Python 3.6+ only
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
#Variables that contains the user credentials to access Twitter API 
consumer_key = os.environ.get('TWITTER_API_KEY')
consumer_secret = os.environ.get('TWITTER_API_SECRET')
access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

#This handles Twitter authetification and the connection to Twitter Streaming API
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
```

##### User Object Data

```
#find user object to get user data
african_influencers_dataset = []
for username in influencer_usernames:
    try:
        search = api.search_users(username) 
        for user in search: 
            if user.screen_name == username:
                africa_influencers_data = []
                africa_influencers_data.extend([user.name,user.screen_name,user.followers_count,user.friends_count,user.statuses_count])
                african_influencers_dataset.extend([africa_influencers_data])
    except:
        continue
```

##### Tweets by the list of users

```
#Original Author: Yabebal Tadesse
def get_tweets_again(self, username, csvfile=None):
        
        
        df = pd.DataFrame(columns=self.cols)
        

        #page attribute in tweepy.cursor and iteration
        for page in tweepy.Cursor(self.api.user_timeline, screen_name=username, count=20, include_rts=False,tweet_mode='extended', since = '2020-05-01').pages(5):

            # the you receive from the Twitter API is in a JSON format and has quite an amount of information attached
            for status in page:
                
                new_entry = []
                status = status._json
                
                #if this tweet is a retweet update retweet count
                if status['created_at'] in df['created_at'].values:
                    i = df.loc[df['created_at'] == status['created_at']].index[0]
                    #
                    cond1 = status['favorite_count'] != df.at[i, 'favorite_count']
                    cond2 = status['retweet_count'] != df.at[i, 'retweet_count']
                    if cond1 or cond2:
                        df.at[i, 'favorite_count'] = status['favorite_count']
                        df.at[i, 'retweet_count'] = status['retweet_count']
                    continue

                #calculate sentiment
                filtered_tweet = self.clean_tweets(status['full_text'])
                blob = TextBlob(filtered_tweet)
                Sentiment = blob.sentiment     
                polarity = Sentiment.polarity
                subjectivity = Sentiment.subjectivity

                new_entry += [status['id'], status['created_at'],
                              status['source'], status['full_text'], filtered_tweet, 
                              Sentiment,polarity,subjectivity, status['lang'],
                              status['favorite_count'], status['retweet_count']]

                new_entry.append(status['user']['screen_name'])

                try:
                    is_sensitive = status['possibly_sensitive']
                except KeyError:
                    is_sensitive = None

                new_entry.append(is_sensitive)

                hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
                new_entry.append(hashtags) #append the hashtags

                #
                mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
                new_entry.append(mentions) #append the user mentions

                try:
                    xyz = status['place']['bounding_box']['coordinates']
                    coordinates = [coord for loc in xyz for coord in loc]
                except TypeError:
                    coordinates = None
                #
                new_entry.append(coordinates)

                try:
                    location = status['user']['location']
                except TypeError:
                    location = ''
                #
                new_entry.append(location)

                #now append a row to the dataframe
                single_tweet_df = pd.DataFrame([new_entry], columns=self.cols)
                df = df.append(single_tweet_df, ignore_index=True)

        #
        df['timestamp'] = df.created_at.map(pd.Timestamp)
        df = df.sort_values('timestamp').set_index('timestamp')
        df = df.drop('id',axis=1)

        if not csvfile is None:
            #save it to file
            df.to_csv(csvfile, index=True,mode='a', encoding="utf-8")
            

        return df
```
 
##### Tweets that mention the list of users

```
#Original Author: Yabebal Tadesse
def get_tweets(self, keyword, csvfile=None):
        
        
        df = pd.DataFrame(columns=self.cols)
        

        #page attribute in tweepy.cursor and iteration
        for page in tweepy.Cursor(self.api.search, q=keyword,count=20, include_rts=False,tweet_mode='extended' , since = '2020-05-01').pages(5):

            # the you receive from the Twitter API is in a JSON format and has quite an amount of information attached
            for status in page:
                
                new_entry = []
                status = status._json
                
                #if this tweet is a retweet update retweet count
                if status['created_at'] in df['created_at'].values:
                    i = df.loc[df['created_at'] == status['created_at']].index[0]
                    #
                    cond1 = status['favorite_count'] != df.at[i, 'favorite_count']
                    cond2 = status['retweet_count'] != df.at[i, 'retweet_count']
                    if cond1 or cond2:
                        df.at[i, 'favorite_count'] = status['favorite_count']
                        df.at[i, 'retweet_count'] = status['retweet_count']
                    continue

                #calculate sentiment
                filtered_tweet = self.clean_tweets(status['full_text'])
                blob = TextBlob(filtered_tweet)
                Sentiment = blob.sentiment     
                polarity = Sentiment.polarity
                subjectivity = Sentiment.subjectivity

                new_entry += [status['id'], status['created_at'],
                              status['source'], status['full_text'], filtered_tweet, 
                              Sentiment,polarity,subjectivity, status['lang'],
                              status['favorite_count'], status['retweet_count']]

                new_entry.append(status['user']['screen_name'])

                try:
                    is_sensitive = status['possibly_sensitive']
                except KeyError:
                    is_sensitive = None

                new_entry.append(is_sensitive)

                hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
                new_entry.append(hashtags) #append the hashtags

                #
                mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
                new_entry.append(mentions) #append the user mentions

                try:
                    xyz = status['place']['bounding_box']['coordinates']
                    coordinates = [coord for loc in xyz for coord in loc]
                except TypeError:
                    coordinates = None
                #
                new_entry.append(coordinates)

                try:
                    location = status['user']['location']
                except TypeError:
                    location = ''
                #
                new_entry.append(location)

                #now append a row to the dataframe
                single_tweet_df = pd.DataFrame([new_entry], columns=self.cols)
                df = df.append(single_tweet_df, ignore_index=True)

        #
        df['timestamp'] = df.created_at.map(pd.Timestamp)
        df = df.sort_values('timestamp').set_index('timestamp')
        df = df.drop('id',axis=1)
        
        if not csvfile is None:
            #save it to file
            df.to_csv(csvfile,mode='a', index=True, encoding="utf-8")
            

        return df
```

### Recap of Data set and features

African leaders list is based on the tweets posted by African leaders on Twitter in response to the COVID-19 pandemic. Influential Twitter users in Africa list are already ranked Twitter influencers in Africa based on their popularity, relevance of their content and size of their audience.

African leaders and influential Twitter users in Africa accounts. This is data extracted from Twitter user objects. African leaders and influential Twitter users in Africa tweets. This is data extracted from the user timelines. African leaders and influential Twitter users in Africa mentions. This is extracted data from tweets that included the usernames of the African leaders and influential Twitter users in Africa. The data is since 1​st May 2020.

### Step 2. Data Cleaning

The data was cleaned as it was being extracted. The data comes in the original form that makes processing hard. The Emoticons were removed as well as consecutive non-ASCII characters. After data extraction there were a few redundancies that were dropped from the data frames obtained.

The data was restructured during collection of data.The time of creation was changed to a timestamp. It is easier to work with. The data types for numerical figures were changed from string to int. Important columns were extracted from the full datasets into small datasets that were used to calculate influence based on the defined criteria.

```
#Original Author: Yabebal Tadesse
def clean_tweets(self, twitter_text):

        #use pre processor
        tweet = p.clean(twitter_text)

         #HappyEmoticons
        emoticons_happy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
            ])

        # Sad Emoticons
        emoticons_sad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
            ])

        #Emoji patterns
        emoji_pattern = re.compile("["
                 u"\U0001F600-\U0001F64F"  # emoticons
                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                 u"\U00002702-\U000027B0"
                 u"\U000024C2-\U0001F251"
                 "]+", flags=re.UNICODE)

        #combine sad and happy emoticons
        emoticons = emoticons_happy.union(emoticons_sad)

        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(tweet)
        #after tweepy preprocessing the colon symbol left remain after      
        #removing mentions
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)

        #replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

        #remove emojis from tweet
        tweet = emoji_pattern.sub(r'', tweet)

        #filter using NLTK library append it to a string
        filtered_tweet = [w for w in word_tokens if not w in stop_words]

        #looping through conditions
        filtered_tweet = []    
        for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
            if w not in stop_words and w not in emoticons and w not in string.punctuation:
                filtered_tweet.append(w)

        return ' '.join(filtered_tweet)
```

### Step 3. Data Analysis

The influencers are classified to identify influence through popularity, reach and relevance. Popularity is obtained from the addition of retweets and likes. Reach is from the difference of the following and the followers. Relevance is the mention. A look at the influencers tweet content has also been done to get an idea of the content they post.

#### Reach

Reach is from the difference of the following and the followers

```
#getting their reach_score = followers-following
african_influencers_dataframe['reach_score']= african_influencers_dataframe['Number of Followers'] - african_influencers_dataframe['Number of Following']
african_leaders_dataframe['reach_score']= african_leaders_dataframe['Number of Followers'] - african_leaders_dataframe['Number of Following']
```

#### Popularity

Popularity is obtained from the addition of retweets and likes.

```
Leaders_Tweets_User['retweet_count'] = pd.to_numeric(Leaders_Tweets_User['retweet_count'])
Leaders_Tweets_User['favorite_count'] = pd.to_numeric(Leaders_Tweets_User['favorite_count'])

Leaders_Tweets_User_Popularity = Leaders_Tweets_User.loc[:,['original_author','retweet_count','favorite_count']]

Influencers_Tweets_User['retweet_count'] = pd.to_numeric(Influencers_Tweets_User['retweet_count'])
Influencers_Tweets_User['favorite_count'] = pd.to_numeric(Influencers_Tweets_User['favorite_count'])

Influencers_Tweets_User_Popularity = Influencers_Tweets_User.loc[:,['original_author','retweet_count','favorite_count']] 

Leaders_Tweets_User_Popularity['popularity_score']= Leaders_Tweets_User_Popularity['retweet_count'] + Leaders_Tweets_User_Popularity['favorite_count']
Influencers_Tweets_User_Popularity['popularity_score']= Influencers_Tweets_User_Popularity['retweet_count'] + Influencers_Tweets_User_Popularity['favorite_count']
```

#### Relevance

Relevance is based on mentions.

```
leader_mentions_count = []
for x in leaders_username:
    line = []
    counter = leader_mentions.count(x)
    line.extend([x,counter])
    leader_mentions_count.append(line)
```

### Step 4. Data Visualization

Visualized the results from the analysis separately first then combined.

For the reach:
```
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top Relevant Leaders on Twitter with reach', fontsize=16)
ax = sns.barplot(x="original_author", y="reach_score", data=top_re_lead)
ax.set_xlabel('Username', fontsize=18)
ax.set_ylabel('Reach Score', fontsize=16)
plt.savefig("img/Top Relevant Leaders on Twitter with reach.png")
plt.show()
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Influencers on Twitter with reach', fontsize=16)
ax = sns.barplot(x="original_author", y="reach_score", data=top_re_infl)
ax.set_xlabel('Username', fontsize=18)
ax.set_ylabel('Reach Score', fontsize=16)
plt.savefig("img/Top 10 Influencers on Twitter with reach.png")
plt.show()
```

#### Top Relevant Leaders on Twitter based on reach

![image](https://user-images.githubusercontent.com/65489309/88452437-59193900-ce67-11ea-8128-88e5c21d6182.png)

#### Top 10 Influencers on Twitter based on reach

![image](https://user-images.githubusercontent.com/65489309/88452432-53235800-ce67-11ea-8579-833f0460e9bf.png)

For the popularity:
```
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Popular Leaders on Twitter', fontsize=16)
ax = sns.barplot(x="original_author", y="popularity_score", data=top_p_lead)
ax.set_xlabel('Username', fontsize=18)
ax.set_ylabel('Popularity Score', fontsize=16)
plt.yscale('log')
plt.show()
plt.savefig("img/Top 10 Popular Leaders on Twitter.png")
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Popular Influencers on Twitter', fontsize=16)
ax = sns.barplot(x="original_author", y="popularity_score", data=top_p_infl)
ax.set_xlabel('Username', fontsize=18)
ax.set_ylabel('Popularity Score', fontsize=16)
plt.yscale('log')
plt.savefig("img/Top 10 Popular Influencers on Twitter.png")
plt.show()
```

#### Top 10 Popular Leaders on Twitter

![image](https://user-images.githubusercontent.com/65489309/88452417-2f601200-ce67-11ea-88e7-3be6c643f4dc.png)

#### Top 10 Popular Influencers on Twitter

![image](https://user-images.githubusercontent.com/65489309/88452420-3424c600-ce67-11ea-9f74-f8ebef2daa67.png)

For the relevance :
```
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Relevant Influencers on Twitter', fontsize=16)
ax = sns.barplot(x="original_author", y="Number of Mentions", data=top_r_infl)
ax.set_xlabel('Username', fontsize=18)
ax.set_ylabel('Relevance Score', fontsize=16)
plt.savefig("img/Top 10 Relevant Influencers on Twitter.png")
plt.show()
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Relevant Leaders on Twitter', fontsize=16)
ax = sns.barplot(x="original_author", y="Number of Mentions", data=top_r_lead)
ax.set_xlabel('Username', fontsize=18)
ax.set_ylabel('Relevance Score', fontsize=16)
plt.yscale('log')
plt.savefig("img/Top 10 Relevant Leaders on Twitter.png")
plt.show()
```

#### Top 10 Relevant Leaders on Twitter

![image](https://user-images.githubusercontent.com/65489309/88452391-fcb61980-ce66-11ea-8cfa-8ef2ca71f880.png)

#### Top 10 Relevant Influencers on Twitter

![image](https://user-images.githubusercontent.com/65489309/88452394-017acd80-ce67-11ea-8bc0-2c8ca9b20a25.png)

Combined data.
```
p_influencers = pd.concat([top_p_infl, top_p_lead], axis=0).sort_values(by=['popularity_score'], ascending=False)
r_influencers = pd.concat([top_r_infl, top_r_lead], axis=0).sort_values(by=['Number of Mentions'], ascending=False)
re_influencers = pd.concat([top_re_infl, top_re_lead], axis=0).sort_values(by=['reach_score'], ascending=False)
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Popular Influencers vs Leaders on Twitter', fontsize=16)
ax = sns.barplot(x="original_author", y="popularity_score", data=p_influencers, hue = 'type')
plt.xticks(rotation=45)
plt.yscale('log')
plt.savefig("img/Top 10 Popular Influencers vs Leaders on Twitter.png")
plt.show()
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Relevant Influencers vs Leaders on Twitter', fontsize=16)
ax = sns.barplot(x="original_author", y="Number of Mentions",hue="type", data=r_influencers)
plt.xticks(rotation=45)
plt.yscale('log')
plt.savefig("img/Top 10 Relevant Influencers vs Leaders on Twitter.png")
plt.show()
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Top 10 Influencers vs Leaders on Twitter with Reach', fontsize=16)
ax = sns.barplot(x="original_author", y="reach_score",hue="type", data=re_influencers)
plt.xticks(rotation=45)
plt.yscale('log')
plt.savefig("img/Top 10 Relevant Influencers vs Leaders on Twitter.png")
plt.show()
```

#### Top 10 Reach Influencers vs Leaders on Twitter

![image](https://user-images.githubusercontent.com/65489309/88452352-a77a0800-ce66-11ea-96be-c1975a8e5828.png)

#### Top 10 Popular Influencers vs Leaders on Twitter

![image](https://user-images.githubusercontent.com/65489309/88452349-a052fa00-ce66-11ea-8a6b-031af172a772.png)

#### Top 10 Relevant Influencers vs Leaders on Twitter

![image](https://user-images.githubusercontent.com/65489309/88452355-acd75280-ce66-11ea-8cca-525cc822ee84.png)


## Conclusions

This activity shows that Trevornoah is influential in Africa. This is because his popularity ranks top. This can be seen in the images above. A retail company should approach him to promote their product if they are entering a space in Africa.
For the full project click on here.

## Future Work

For extension of the identification of the influencers with the most influence and rank them accordingly, one could change relevance from mentions alone to mentions and comments. To see if there is a difference. Influencers who have promoted a business through their platform should be identified to see their impact on the business.

## References

Social Media Stats Africa | StatCounter Global Stats. (2020). Retrieved 19 July 2020, from https://gs.statcounter.com/social-media-stats/all/africa
Chaffey, D., & Smith, P. (2013). Emarketing Excellence: Planning and optimizing your digital marketing (4th ed.). New York: Routledge.
