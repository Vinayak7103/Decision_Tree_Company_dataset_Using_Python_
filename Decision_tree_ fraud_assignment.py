#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[2]:


fraud= pd.read_csv('C:/Users/vinay/Downloads/Fraud_check.csv')


# In[3]:


fraud


# In[4]:


fraud.rename(columns = {'Taxable.Income':'Taxable_Income','City.Population':'City_Population','Work.Experience':'Work_Experience','Marital.Status':'Marital_Status'},inplace = True) ;fraud


# In[5]:


fraud1 = pd.get_dummies(fraud,columns=['Undergrad','Marital_Status','Urban']);fraud1


# In[6]:


fraud1["income"]="<=30000"
fraud1.loc[fraud1["Taxable_Income"]>=30000,"income"]="Good"
fraud1.loc[fraud1["Taxable_Income"]<=30000,"income"]="Risky"


# In[7]:


fraud1


# In[8]:


fraud1.drop(["Taxable_Income"],axis=1,inplace=True)


# In[9]:


fraud1


# In[10]:


label_encoder = preprocessing.LabelEncoder()
fraud1['income']= label_encoder.fit_transform(fraud1['income']) 


# In[11]:


fraud1


# In[12]:


fraud1['income'].unique()


# In[13]:


fraud1.income.value_counts()


# In[14]:


colnames = list(fraud1.columns)
colnames


# In[15]:


x=fraud1.iloc[:,0:9]
y=fraud1['income']


# In[16]:


x


# In[17]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[18]:


x_train


# ## Building Decision Tree Classifier using Entropy Criteria

# In[19]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[20]:


tree.plot_tree(model)


# In[21]:


fn=['City_Population','Work_Experience','Undergrad_NO','Undergrad_YES','Marital_Status_Divorced','Marital_Status_Married','Marital_Status_Single','Urban_NO','Urban_YES']
cn=['Good','Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[22]:


#Predicting on test data
preds = model.predict(x_test) 
pd.Series(preds).value_counts()


# In[23]:


preds


# In[24]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[25]:


# Accuracy 
np.mean(preds==y_test)


# ## Building Decision Tree Classifier (CART) using Gini Criteria

# In[26]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[27]:


model_gini.fit(x_train, y_train)


# In[28]:


#Prediction and computing the accuracy
pred=model_gini.predict(x_test)
np.mean(preds==y_test)


# ## Decision Tree Regression Example

# In[29]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[30]:


array = fraud1.values
X = array[:,0:9]
y = array[:,9]


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)


# In[32]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[34]:


#Find the accuracy
model.score(X_test,y_test)


# In[ ]:




