#!/usr/bin/env python
# coding: utf-8

# # Handling Null Values

# #import libraries

# In[1]:


import numpy as np


# In[3]:


import pandas as pd


# #loading dataset

# In[11]:


dataset=pd.read_csv('C:/Users/ISHITA SWAMI/Downloads/people_data.csv')


# In[12]:


print(dataset)


# #Check the num of null values

# In[13]:


print(dataset.isnull().sum())


# #Drop the missing values

# In[21]:


print(dataset.dropna(inplace=True))


# #print the dataset

# In[22]:


print(dataset)

# #Dependent And Independent Vector

# In[30]:


x1=dataset.iloc[:,:-1].values


# In[31]:


y1=dataset.iloc[:,-1].values


# In[32]:


print(x1)


# In[33]:


print(y1)


# # Data Encoding

# #one hot encoding: Convert city text to binary values

# In[23]:


from sklearn.compose import ColumnTransformer


# In[24]:


from sklearn.preprocessing import OneHotEncoder


# In[25]:


ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")


# In[27]:


x=np.array(ct.fit_transform(x1))


# In[28]:


print(x)





# #Label Encoding

# In[35]:


from sklearn.preprocessing import LabelEncoder


# In[37]:


le=LabelEncoder()


# In[39]:


y=le.fit_transform(y1)


# In[40]:


print(y)


# # Splitting

# In[41]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)


# In[44]:


print(x_train)


# In[45]:


print(x_test)


# In[46]:


print(y_train)


# In[47]:


print(y_test)


# # Scaling

# #Standardization & Normalization

# In[48]:


from sklearn.preprocessing import StandardScaler


# In[49]:


scaler=StandardScaler()


# In[50]:


x_train[:,4:]=scaler.fit_transform(x_train[:,4:])


# In[51]:


x_test[:,4:]=scaler.fit_transform(x_test[:,4:])


# In[52]:


print(x_train)


# In[53]:


print(x_test)


# In[ ]:




