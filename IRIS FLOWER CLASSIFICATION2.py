#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DataFlair Iris Flower Classification
# Import Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


# In[2]:


df=pd.read_csv("Iris.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


#checking for null values
df.isnull().sum()


# In[8]:


df.columns


# In[9]:


#Drop unwanted columns
df=df.drop(columns="Id")


# In[10]:


df


# In[11]:


df['Species'].value_counts()


# In[12]:


sns.countplot(df['Species']);


# In[13]:


x=df.iloc[:,:4]
y=df.iloc[:,4]


# In[14]:


x


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


y_train.shape


# In[20]:


y_test.shape


# In[21]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[22]:


model.fit(x_train,y_train)


# In[23]:


y_pred=model.predict(x_test)


# In[24]:


y_pred


# In[25]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[26]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:




