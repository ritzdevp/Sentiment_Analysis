
# coding: utf-8

# In[197]:


from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


# Importing data from disk

# In[173]:


data_file = 'C:\\Users\\SONY\\Downloads\\Amazon_Unlocked_Mobile.csv'
data = pd.read_csv( data_file)


# In[174]:


data.head()


# In[175]:


product_name = []
for item in data["Product Name"]:
    if (item in product_name):
        continue
    else:
        product_name.append(item)
        


# In[176]:


len(product_name) # 4410 phones


# There are 4410 phone models in this data set.

# In[177]:


data["Brand Name"]
brands = []
for item in data["Brand Name"]:
    if (item in brands):
        continue
    else:
        brands.append(item)


# In[178]:


len(brands) 


# There are 385 brands in this data set.

# Putting the data in a Pandas Dataframe.

# In[179]:


data_df = pd.DataFrame(data) #converting the data into a pandas dataframe.


# In[181]:


data_df.head()


# In[183]:


data_df = shuffle(data_df) #Shuffle Data 


# In[184]:


data_df[:10]


# Cleaning data by removing rows having 'null' values.

# In[185]:


#dropped rows having NaN values
data_df = data_df.dropna()


# Taking 75 % of the data set as training data and 25 % as test data.

# In[187]:


data_df_train = data_df[:(int)(0.75 * len(data_df))]
len(data_df_train)


# In[188]:


data_df_test = data_df[250751:]
len(data_df_test)


# In[228]:


data_df.describe() # General Description of data_df


# Top 10 brands in the data set having maximum Rating.

# In[293]:


testing = pd.pivot_table(data_df,index=['Brand Name'],values=['Rating', 'Review Votes'],
               columns=[],aggfunc=[np.sum, np.mean],fill_value=0)
testing = testing.sort_values(by=('sum', 'Rating'), ascending = False)
testing.head(10)


# In[241]:


#Trying to find relation between price & rating 


# In[261]:


import matplotlib.pyplot as plt
ylabel = data_df["Price"]
plt.ylabel("Price")
plt.xlabel("Rating")
xlabel = data_df["Rating"]
plt.scatter(xlabel, ylabel, alpha=0.1)
plt.show()


# In[260]:


ylabel2 = data_df["Price"]
plt.ylabel("Price")
xlabel2 = data_df["Review Votes"]
plt.xlabel("Review Votes")
plt.scatter(xlabel2, ylabel2, alpha=0.1)
plt.show()


# In[262]:


ylabel3 = data_df["Rating"]
plt.ylabel("Rating")
xlabel3 = data_df["Review Votes"]
plt.xlabel("Review Votes")
plt.scatter(xlabel3, ylabel3, alpha=0.1)
plt.show()


# In[264]:


corr_matrix = data_df.corr()
corr_matrix["Rating"].sort_values(ascending = False)


# It is observed that Rating has a NEGATIVE CORRELATION with Review Votes = -0.046526

# In[265]:


corr_matrix = data_df.corr()
corr_matrix["Price"].sort_values(ascending = False)


# It is observed that Rating has a POSITIVE CORRELATION with Price = 0.073948
