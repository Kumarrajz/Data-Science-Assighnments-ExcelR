#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install selenium')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver


# In[5]:


pip install wordcloud


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver


# In[7]:


macbook_air=[]


# In[8]:


for i in range (1,41):
    mac=[]
    url="https://www.amazon.in/Apple-MacBook-Chip-13-inch-256GB/product-reviews/B08N5W4NNB/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviewsshowViewpoints=1&pageNumber="+str(i)
    response=requests.get(url)
    soup=bs(response.content,"html.parser")
    reviews=soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})
    for i in range (len(reviews)):
        mac.append(reviews[i].text)
    macbook_air=macbook_air+mac   


# In[9]:


macbook_air


# In[10]:


len(macbook_air)


# In[11]:


rev={"review":macbook_air}


# In[12]:


review_data=pd.DataFrame.from_dict(rev)
pd.set_option('max_colwidth',800)


# In[13]:


review_data


# In[14]:


text=" ".join(review_data)


# In[15]:


def clean_text(text):
    text=re.sub('@[A-Za-z0-9]+','',str(text))#To remove @
    text=re.sub('#','',str(text))#To remove #
    text=re.sub('RT[\s]+','',str(text))#To remove retweets
    text=re.sub('\n\n','',str(text))  #To remove \n
    text=text.lower()
    text=re.sub('https?:\/\/\S+','',str(text)) #To remove links
    
    return text
review_data["review"]=review_data["review"].apply(clean_text)


# In[16]:


review_data


# In[17]:


text=" ".join(review_data["review"])


# In[21]:


def clean_text(text):
    text=re.sub('@[A-Za-z0-9]+','',str(text))#To remove @
    text=re.sub('#','',str(text))#To remove #
    text=re.sub('RT[\s]+','',str(text))#To remove retweets
    text=re.sub('\n\n','',str(text))  #To remove \n
    text=text.lower()
    text=re.sub('https?:\/\/\S+','',str(text)) #To remove links
    
    return text
review_data["review"]=review_data["review"].apply(clean_text)


# In[23]:


review_data


# In[24]:


text=" ".join(review_data["review"])


# In[25]:


text


# In[27]:


import nltk


# In[29]:


from nltk import word_tokenize


# In[34]:


from nltk.tokenize import word_tokenize


# In[39]:


text_tokens = word_tokenize(text)


# In[38]:


nltk.download('punkt')


# In[40]:


text_tokens


# In[43]:


text_without_sw=[word for word in text_tokens if not word in stopwords.words()]


# In[42]:


nltk.download('stopwords')


# In[44]:


tf=TfidfVectorizer()


# In[45]:


text_tf=tf.fit_transform(text_without_sw)


# In[46]:


feature_names=tf.get_feature_names()
dense=text_tf.todense()
denselist=dense.tolist()
df=pd.DataFrame(denselist,columns=feature_names)


# In[48]:


df


# In[ ]:





# In[49]:


words_list=" ".join(df)


# In[50]:


wordcloud=WordCloud(background_color="black",width=1800,height=1300).generate(words_list)
plt.imshow(wordcloud)


# In[52]:


with open ("C:\XboxGames\positive-words.txt","r") as pw:
    positive_words=pw.read().split("/n")
    
positive_words=positive_words[35:]


# In[56]:


neg_text=" ".join([word for word in df if not word in negative_words])


# In[55]:


with open ("C:\XboxGames\positive-words.txt","r") as nw:
    negative_words=nw.read().split("/n")
    
negative_words=negative_words[35:]    


# In[57]:


pos_text=" ".join([word for word in df if not word  in positive_words])


# In[58]:


pos_wordcloud=WordCloud(background_color="black",width=1800,height=1400).generate(pos_text)
plt.imshow(pos_wordcloud)


# In[59]:


neg_wordcloud=WordCloud(background_color="black",width=1800,height=1400).generate(neg_text)
plt.imshow(neg_wordcloud)


# In[66]:


from textblob import TextBlob
def sentiment_analysis(ds):
    sentiment = TextBlob(ds["review"]).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])

review_data[["subjectivity", "polarity"]] = review_data.apply(sentiment_analysis, axis=1)
review_data


# In[65]:


from textblob import TextBlob


# In[67]:


def analysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"
    
review_data["analysis"] = review_data["polarity"].apply(analysis)
review_data


# In[68]:


review_data['analysis'].value_counts()


# In[69]:


#positive comments
((228+33)/280)*100


# In[70]:


#negative comments
(19/199)*100


# In[ ]:


so in this review data only 9% is about the negative comments so the macbook air product Received good product reviews from users

