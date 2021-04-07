#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import tree


# In[3]:


company= pd.read_csv('C:/Users/vinay/Downloads/Company_Data.csv')


# In[4]:


company.head()


# In[5]:


company["Sales"].min()


# In[6]:


company["Sales"].max()


# In[7]:


company["Sales"].value_counts()


# ## Checking for maximum and minimum values to decide what will be the cut off point

# In[8]:


np.median(company["Sales"])


# In[11]:


##Knowing the middle value by looking into median so that i find the middle value to check to divide data into two levels.
company["sales"]= "<=7.49"
company.loc[company["Sales"]>=7.49,"sales"]=">=7.49"


# In[12]:


company["sales"].unique()
company["sales"].value_counts()


# In[13]:


##Dropping Sales column from the data 
company.drop(["Sales"],axis=1,inplace = True)


# In[14]:


company.isnull().sum() # no null value


# In[16]:


#As, the fit does not consider the String data, we need to encode the data.


# In[17]:


from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
for column_name in company.columns:
    if company[column_name].dtype == object:
        company[column_name] = le.fit_transform(company[column_name])
    else:
        pass


# In[48]:


features = company.iloc[:,0:10] 
labels = company.iloc[:,10]


# In[20]:


##Splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.3,stratify = labels)


# In[21]:


y_train.value_counts()
y_test.value_counts()


# In[22]:


#Building the model
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[23]:


model = DT(criterion='entropy') 
model.fit(x_train,y_train)


# In[24]:


##prediction on Training data
pred_train = pd.DataFrame(model.predict(x_train))


# In[25]:


##Finding Accuracy for train data
acc_train = accuracy_score(y_train,pred_train)


# In[27]:


acc_train #100%


# In[28]:


## Confusion matrix
confusion_mat = pd.DataFrame(confusion_matrix(y_train,pred_train,))


# In[29]:


##prediction on test data
pred_test = pd.DataFrame(model.predict(x_test))


# In[30]:


##accuracy on test data
acc_test = accuracy_score(y_test,pred_test)


# In[31]:


acc_test ##70%


# In[32]:


#Confusion matrix
confusion_test = pd.DataFrame(confusion_matrix(y_test,pred_test))


# In[33]:


confusion_test


# # Building Decision Tree Classifier using Entropy Criteria

# In[53]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=2)
model.fit(x_train,y_train)


# In[54]:


tree.plot_tree(model)


# In[61]:


fn=['City_Population','Work_Experience','Undergrad_NO','Undergrad_YES','Marital_Status_Divorced','Marital_Status_Married','Marital_Status_Single','Urban_NO','Urban_YES']
cn=['Good','Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[62]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts()


# In[64]:


preds


# In[63]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[66]:


# Accuracy 
np.mean(preds==y_test)


# ## Building Decision Tree Classifier (CART) using Gini Criteria

# In[67]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[71]:


model_gini.fit(x_train, y_train)


# In[72]:


#Prediction and computing the accuracy
pred=model_gini.predict(x_test)
np.mean(preds==y_test)


# ## Decision Tree Regression Example

# In[73]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[75]:


array = company.values
X = array[:,0:9]
y = array[:,9]


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)


# In[77]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[78]:


#Find the accuracy
model.score(X_test,y_test)


# In[ ]:




