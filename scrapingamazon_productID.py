#!/usr/bin/env python
# coding: utf-8
import requests 
import pandas as pd
from bs4 import BeautifulSoup 


# In[33]:
referer_link = 'https://www.amazon.com/s?k=turtles+climate+adventure&crid=2KGU6WPHWFDL7&sprefix=turtles+climate+adventure%2Caps%2C137&ref=nb_sb_noss_1'

num_review_pages = 2
# In[35]:


# In[40]:


header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36','referer':referer_link}

# In[45]:


cookie={} # insert request cookies within{}

# In[46]:


def Searchasin(asin):
    url="https://www.amazon.com/dp/"+asin
    print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    if page.status_code==200:
        return page
    else:
        return "Error"
# ## function to pass on the link of 'see all reviews' and extract the content

# In[47]:
def Searchreviews(review_link):
    url="https://www.amazon.com"+review_link
    print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    if page.status_code==200:
        return page
    else:
        return "Error"


link=[]
response=Searchasin('B0BGN8XXW6')
soup=BeautifulSoup(response.content, "html.parser")
for i in soup.findAll("a",{'data-hook':"see-all-reviews-link-foot"}):
    link.append(i['href'])


# In[58]:


len(link)


# #### Now we have the 'see all review' links. Using this link along with a page number, we can extract the reviews in any number of pages for all the products

# In[71]:


link


# In[59]:


reviews=[]
for j in range(len(link)):
    for k in range(num_review_pages):
        response=Searchreviews(link[j]+'&pageNumber='+str(k))
        soup=BeautifulSoup(response.content, "html.parser")
        for i in soup.findAll("span",{'data-hook':"review-body"}):
            reviews.append(i.text)


# In[60]:


len(reviews)


# In[61]:


rev={'reviews':reviews}


# In[64]:


review_data=pd.DataFrame.from_dict(rev)
pd.set_option('max_colwidth',800)


# In[72]:


review_data.head(5)


# In[66]:


#review_data.shape


# In[78]:


review_data.to_csv('Scraping reviews.csv') #converting the dataframe to a csv file so as to use it later for further analysis


# In[ ]:




