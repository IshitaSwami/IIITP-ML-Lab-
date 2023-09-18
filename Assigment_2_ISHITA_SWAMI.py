#!/usr/bin/env python
# coding: utf-8

# # Ordinal Encoding   #Hash Encoding   #Frequency Encoding

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns


# In[19]:


dataset=pd.read_csv('C:/Users/ISHITA SWAMI/Downloads/people_data.csv')


# In[25]:


print(dataset)


# In[29]:

# Ordinal Encoding 

from sklearn.preprocessing import OrdinalEncoder


# In[30]:


ord1=OrdinalEncoder()


# In[31]:


ord1.fit([dataset['Purchased']])


# In[33]:


dataset["Purchased"]= ord1.fit_transform(dataset[["Purchased"]])


# In[34]:


dataset.head(10)
print(dataset)


# In[37]:

 #Hash Encoding 

from sklearn.feature_extraction import FeatureHasher


# In[38]:


h = FeatureHasher(n_features = 3, input_type ='string')


# In[40]:


hashed_Feature = h.fit_transform(dataset['Country'])


# In[41]:


hashed_Feature = hashed_Feature.toarray()


# In[42]:


dataset = pd.concat([dataset, pd.DataFrame(hashed_Feature)], axis = 1)


# In[43]:


dataset.head(10)
print(dataset)

# In[45]:

#Frequency Encoding

fq = dataset.groupby('Country').size()/len(dataset)  


# In[46]:


dataset.loc[:, "{}_freq_encode".format('Country')] = dataset['Country'].map(fq)  


# In[47]:


dataset = dataset.drop(['Country'], axis = 1) 


# In[48]:



fq.plot.bar(stacked = True)  


# In[50]:


dataset.head(10)
print(dataset)


# In[ ]:

from category_encoders import BinaryEncoder
encoder = BinaryEncoder(cols =['Purchased'])

newdata = encoder.fit_transform(dataset['Purchased'])

dataset = pd.concat([dataset, newdata], axis = 1)

dataset = dataset.drop(['Purchased'], axis = 1)
dataset.head(10)
print(dataset)




# In[ ]:





# In[ ]:




