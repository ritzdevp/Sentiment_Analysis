
# coding: utf-8

# # Sentiment Analysis on Amazon Unlocked Mobile Phones Using NLTK

# In[388]:


#Importing libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import nltk


# Importing data from disk
# ##### link: https://www.kaggle.com/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones/data

# In[390]:


data_file = 'C:\\Users\\SONY\\Downloads\\Amazon_Unlocked_Mobile.csv'
#reading csv file
data = pd.read_csv( data_file)


# In[391]:


data.head() 


# In[392]:


product_name = []
for item in data["Product Name"]:
    if (item in product_name):
        continue
    else:
        product_name.append(item)
        


# In[393]:


len(product_name) # 4410 phones


# ### There are 4410 phone models in this data set.

# In[394]:


data["Brand Name"]
brands = []
for item in data["Brand Name"]:
    if (item in brands):
        continue
    else:
        brands.append(item)


# In[395]:


len(brands) 


# ### There are 385 brands in this data set.

# Putting the data in a Pandas Dataframe.

# In[396]:


data_df = pd.DataFrame(data) #converting the data into a pandas dataframe.


# In[397]:


data_df.head()


# In[398]:


data_df = shuffle(data_df) #Shuffle Data 


# In[141]:


data_df[:10]


# #### Cleaning data by removing rows having 'null' values.

# In[142]:


#dropped rows having NaN values
data_df = data_df.dropna()


# In[399]:


# General Description of data_df
data_df.describe() 


# ### Top 10 brands in the data set sorted on the basis of sum of Ratings.

# In[401]:


info = pd.pivot_table(data_df,index=['Brand Name'],values=['Rating', 'Review Votes'],
               columns=[],aggfunc=[np.sum, np.mean],fill_value=0)
info = info.sort_values(by=('sum', 'Rating'), ascending = False)

info.head(10)


# ### CoRelation between price & rating 

# In[402]:


import matplotlib.pyplot as plt
ylabel = data_df["Price"]
plt.ylabel("Price")
plt.xlabel("Rating")
xlabel = data_df["Rating"]
plt.scatter(xlabel, ylabel, alpha=0.1)
plt.show()


# ### CoRelation between Price and Review Votes

# In[147]:


ylabel2 = data_df["Price"]
plt.ylabel("Price")
xlabel2 = data_df["Review Votes"]
plt.xlabel("Review Votes")
plt.scatter(xlabel2, ylabel2, alpha=0.1)
plt.show()


# #### Strong co-relation between review votes and price.

# ### CoRelation between Rating and Review Votes

# In[148]:


ylabel3 = data_df["Rating"]
plt.ylabel("Rating")
xlabel3 = data_df["Review Votes"]
plt.xlabel("Review Votes")
plt.scatter(xlabel3, ylabel3, alpha=0.1)
plt.show()


# In[149]:


corr_matrix = data_df.corr()
corr_matrix["Rating"].sort_values(ascending = False)


# #### It is observed that Rating has a NEGATIVE CORRELATION with Review Votes = -0.046526

# In[150]:


corr_matrix = data_df.corr()
corr_matrix["Price"].sort_values(ascending = False)


# #### It is observed that Rating has a POSITIVE CORRELATION with Price = 0.073948

# In[151]:


all_reviews = data_df["Reviews"]
all_reviews.head()


# #### Reset index (post-shuffling)

# In[403]:


#reset_index
data_df = data_df.reset_index(drop=True)


# In[404]:


data_df.head()


# ### NLTK function to find sentiment value and sentiment.

# In[155]:


all_reviews = data_df['Reviews']
all_sent_values = []
all_sentiments = []


# In[739]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
def sentiment_value(paragraph):
    analyser = SentimentIntensityAnalyzer()
    result = analyser.polarity_scores(paragraph)
    score = result['compound']
    return round(score,1)


# In[743]:


sample = data_df['Reviews'][1231]
print(sample)
print('Sentiment: ')
print(sentiment_value(sample))


# In[750]:


sample1 = data_df['Reviews'][99314]
print(sample1)
print('Sentiment: ')
print(sentiment_value(sample1))


# In[755]:


sample2 = data_df['Reviews'][9001]
print(sample2)
print('Sentiment: ')
print(sentiment_value(sample2))


# #### Problem: Calling the function for full set of reviews ie: for 400,000+ , takes time. (8 minutes for 20000 rows in my system.)

# In[406]:


for i in range(0,20000):
    all_sent_values.append(sentiment_value(all_reviews[i])) # 8 minutes for calculation 


# In[158]:


len(all_sent_values)


# In[408]:


#Sentiment Analysis on first 20,000 rows
temp_data = data_df[0:20000]


# In[409]:


temp_data.shape


# ### Intervals
# #### [ -1, -0.5) : 1, V.Negative
# #### [-0.5, 0) : 2, Negative
# #### [0] : 3, Neutral
# #### (0, 0.5) : 4, Positive
# #### [0.5, 1] : 5, V.Positive
# 

# In[410]:


SENTIMENT_VALUE = []
SENTIMENT = []
for i in range(0,20000):
    sent = all_sent_values[i]
    if (sent<=1 and sent>=0.5):
        SENTIMENT.append('V.Positive')
        SENTIMENT_VALUE.append(5)
    elif (sent<0.5 and sent>0):
        SENTIMENT.append('Positive')
        SENTIMENT_VALUE.append(4)
    elif (sent==0):
        SENTIMENT.append('Neutral')
        SENTIMENT_VALUE.append(3)
    elif (sent<0 and sent>=-0.5):
        SENTIMENT.append('Negative')
        SENTIMENT_VALUE.append(2)
    else:
        SENTIMENT.append('V.Negative')
        SENTIMENT_VALUE.append(1)
        
        


# In[411]:


#update to temp_data


# In[412]:


temp_data['SENTIMENT_VALUE'] = SENTIMENT_VALUE
temp_data['SENTIMENT'] = SENTIMENT


# In[413]:


temp_data.head()


# ##### Accuracy

# In[187]:


#find accuracy
counter = 0
for i in range(0,20000):
    if (abs(temp_data['Rating'][i]-temp_data['SENTIMENT_VALUE'][i])>1):
        counter += 1
    


# In[188]:


counter


# ###### 4570 occurences where Rating and Sentiment differ by more than 1.

# In[189]:


accuracy = (temp_data.shape[0]-counter)/temp_data.shape[0]


# In[414]:


percent_accuracy = accuracy*100


# In[415]:


percent_accuracy


# ### 77.15 % equal values of Rating and Sentiment Values (+/- 1) 

# In[416]:


temp_data.head()


# In[451]:


xaxis = []
for i in range(0,20000):
    xaxis.append(i)

ylabel_new_1 = all_sent_values[:20000]

xlabel = xaxis
plt.figure(figsize=(9,9))
plt.xlabel('ReviewIndex')
plt.ylabel('SentimentValue(-1 to 1)')
plt.plot(xlabel, ylabel_new_1, 'ro',  alpha=0.04)

plt.title('Scatter Intensity Plot of Sentiments')
plt.show()


# ### Observation: Sentiment variation is concentrated towards positivity.

# In[733]:


product_name_20k = []
for item in temp_data["Product Name"]:
    if (item in product_name_20k):
        continue
    else:
        product_name_20k.append(item)


# In[279]:


len(product_name_20k)


# 2245 different products in temp_data set.

# ###### For first 20,000

# In[282]:


brands_temp = []
for item in temp_data["Brand Name"]:
    if (item in brands_temp):
        continue
    else:
        brands_temp.append(item)


# In[283]:


len(brands_temp)


# 221 brands in the set.

# In[453]:


testing2 = pd.pivot_table(temp_data,index=['Brand Name'],values=['Rating', 'Review Votes','SENTIMENT_VALUE'],
               columns=[],aggfunc=[np.sum, np.mean],fill_value=0)
testing2 = testing2.sort_values(by=('sum', 'Rating'), ascending = False)
testing2.head(10)


# #### Top 10 Brand names.
# ##### Samsung
# ##### BLU
# ##### Apple
# ##### LG
# ##### Nokia
# ##### BlackBerry
# ##### Motorola
# ##### HTC
# ##### CNPGD
# ##### OtterBox
# ### Rating and Sentiment Value data are accurate with respect to each other.

# #### Top Phone Models

# In[288]:


testing3 = pd.pivot_table(temp_data,index=['Product Name'],values=['Rating', 'Review Votes','SENTIMENT_VALUE'],
               columns=[],aggfunc=[np.sum, np.mean],fill_value=0)
testing3 = testing3.sort_values(by=('sum', 'Rating'), ascending = False)
testing3.head(10)


# ### Sum and Mean Plots of Rating with Sentiments for first 20,000 rows.

# In[454]:


import pylab

names = testing2.index[:10]
y = testing2['sum', 'SENTIMENT_VALUE'][:10]
y2 = testing2['sum', 'Rating'][:10]



pylab.figure(figsize=(15,7))
x = range(10)
pylab.subplot(2,1,1)
pylab.xticks(x, names)
pylab.ylabel('Summed Values')
pylab.title('Total Sum Values')
pylab.plot(x,y,"r-",x,y2,'b-')
pylab.legend(['SentimentValue', 'Rating'])

y_new = testing2['mean', 'SENTIMENT_VALUE'][:10]
y2_new = testing2['mean', 'Rating'][:10]



pylab.figure(figsize=(15,7))


pylab.subplot(2,1,2)
pylab.xticks(x, names)
pylab.ylabel('Mean Values')
pylab.title('Mean Values')
pylab.plot(x,y_new,"r-",x,y2_new,'b-')
pylab.legend(['SentimentValue', 'Rating'])


pylab.show()


# ### Sentiment Analysis For Top 5 brands

# In[546]:


samsung = []
blu = []
apple = []
lg = []
nokia = []



for i in range(0,20000):
    score = all_sent_values[i]
    brand = temp_data['Brand Name'][i]
    if (brand == 'Samsung'):
        samsung.append(score)
    elif (brand == 'BLU'):
        blu.append(score)
    elif (brand == 'Apple'):
        apple.append(score)
    elif (brand == 'LG'):
        lg.append(score)
    elif (brand == 'Nokia'):
        nokia.append(score)
    else:
        continue


# In[550]:


list_of_brands = [samsung, blu, apple,lg,nokia]
name_of_brands = ['Samsung', 'BLU', 'Apple', 'LG', 'Nokia']


# In[722]:


def plot_brand(brand, name):
    pylab.figure(figsize=(20,3))
    x = range(0,800)
    
    #pylab.xticks(x)
    pylab.ylabel('Sentiment')
    pylab.title(name)
    #pylab.plot(x,brand,"ro", alpha = 0.2)
    pylab.plot(x, brand[:800], color='#4A148C', linestyle='none', marker='o',ms=9, alpha = 0.4)
    
    pylab.show()


# In[723]:


for i in range(0,len(list_of_brands)):
    plot_brand(list_of_brands[i],name_of_brands[i])


# ## Observation : 
# #### 1. Sentiment concentration towards positivity decreases as we move from top to lower brands.
# #### 2. Population towards negativity and neutrality keeps on increasing as we move downwards.
