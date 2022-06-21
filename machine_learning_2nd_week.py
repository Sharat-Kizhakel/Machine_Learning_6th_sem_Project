#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:
# import driver as d

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns


# In[2]:


df = pd.read_csv("data.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.shape


# In[9]:


df = df.dropna(axis=1)


# In[12]:


df.shape
df.describe()


# In[14]:


df['diagnosis'].value_counts()


# In[16]:


# count of malignant and begnign cells
sns.countplot(df['diagnosis'], label="count")


# In[19]:


# label encoding
labelencoder_Y = LabelEncoder()
df.iloc[:, 1] = labelencoder_Y.fit_transform(df.iloc[:, 1].values)
df.head()


# In[21]:


sns.pairplot(df.iloc[:, 1:5], hue="diagnosis")


# In[22]:


df.iloc[:, 1:32].corr()


# In[34]:


# heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(df.iloc[:, 1:10].corr(), annot=True, fmt=".0%")


# In[25]:


X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values


# In[26]:


# splitting data set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0)


# In[28]:


X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# In[31]:


# function for 3 supervised learning algorithms to print training accuracy

def models(X_train, Y_train):
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=0, criterion="entropy")
    tree.fit(X_train, Y_train)

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(
        random_state=0, criterion="entropy", n_estimators=10)
    forest.fit(X_train, Y_train)

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    print('1. Decision Tree accuracy:', tree.score(X_train, Y_train))
    print('2. Random Forest accuracy:', forest.score(X_train, Y_train))
    print('3. Logistic Regression accuracy:', log.score(X_train, Y_train))

    return tree, forest, log


# In[30]:

# stores accuracies of the models
model = models(X_train, Y_train)


# Test results

# In[32]:

ml_algo = ['Decision Tree', 'Random Forest', 'Logistic Regression']


# def accuracy_display(classifier_name):
#     for i in range(len(model)):
#         if classifier_name == "Logistic Regression":
#             st.write(model.index('Logistic Regression'))
#             st.write("Model:", classifier_name)
#             st.write("Accuracy:", accuracy_score(
#                 Y_test, model[2].predict(X_test)))
#         elif classifier_name == "Decision Tree":
#             st.write("Accuracy:", accuracy_score(
#                 Y_test, model[0].predict(X_test)))
#             st.write("Model:", classifier_name)
#         elif classifier_name == "Random Forest":
#             st.write("Accuracy:", accuracy_score(
#                 Y_test, model[1].predict(X_test)))
#             st.write("Model:", classifier_name)
# st.write(classification_report(Y_test, model[i].predict(X_test)))

# st.write("")
# st.write("")
# st.write("")


# In[33]:


pred = model[2].predict(X_test)
print('Predicted values:')
print(pred)
print('Actual values:')
print(Y_test)
