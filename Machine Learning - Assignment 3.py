#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[42]:


import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[43]:


df=pd.read_csv("~/Downloads/Dataset/train.csv")


# In[44]:


df.head()


# In[45]:


#correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case in class.
le = preprocessing.LabelEncoder()
df['Sex'] = le.fit_transform(df.Sex.values)
df['Survived'].corr(df['Sex'])


# In[46]:


#a. Do you think we should keep this feature?
#No as the accuracy is 54% 


# In[47]:


matrix = df.corr()
print(matrix)


# In[48]:


df.corr().style.background_gradient(cmap="Greens")


# In[49]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[52]:


#Naïve Bayes method

train_raw = pd.read_csv("~/Downloads/Dataset/train.csv")
test_raw = pd.read_csv("~/Downloads/Dataset/test.csv")

# Join data to analyse and process the set as one.
train_raw['train'] = 1
test_raw['train'] = 0
df = train_raw.append(test_raw, sort=False)




features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

df = df[features + [target] + ['train']]
# Categorical values need to be transformed into numeric.
df['Sex'] = df['Sex'].replace(["female", "male"], [0, 1])
df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = df.query('train == 1')
test = df.query('train == 0')


# In[53]:



# Drop missing values from the train set.
train.dropna(axis=0, inplace=True)
labels = train[target].values


train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)


# In[54]:


from sklearn.model_selection import train_test_split, cross_validate

X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)


# In[55]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
# Suppress warnings
warnings.filterwarnings("ignore")


# In[56]:


classifier = GaussianNB()

classifier.fit(X_train, Y_train)


# In[98]:


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# # Question 2

# In[57]:


glass=pd.read_csv("~/Downloads/Dataset/glass.csv")


# In[58]:


glass.head()


# In[59]:


glass.corr().style.background_gradient(cmap="Greens")


# In[60]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[61]:


features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'


X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass['Type'],test_size=0.2, random_state=1)

classifier = GaussianNB()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[62]:



from sklearn.svm import SVC, LinearSVC

classifier = LinearSVC()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[ ]:


#Which algorithm you got better accuracy? Can you justify why?

#Navie Bayes algorithm got better accuracy than SVM algorithm. Navie Bayes requires a small amount of training data, 
#It tends to perform well for problems like spam detection and text classification.
#SVM algorithm typically provides output easily interpretable probabilities.
#svm is more expensive then navie bayes.





