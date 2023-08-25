#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys


# In[3]:


import joblib

sys.modules['sklearn.externals.joblib'] = joblib


# In[4]:


def predict(data):
 joblib_knn_model = joblib.load(joblib_file)
 return joblib_knn_model.predict(data)

