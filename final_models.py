#!/usr/bin/env python
# coding: utf-8

# ### Co-authored by Tanya, Bhavik, Mudit, Srihit and Jaykumar as a result of our ME781 Data Mining final project.

# Based on Data set from UCI's Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier


# In[3]:


pip install xgboost


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


cd "/content/drive/My Drive/Colab Notebooks"


# ## Reading Data

# In[5]:


# Reading shopping data
X_train = pd.read_csv('ShoppingData.csv')
df = X_train.copy()
df.head()


# ## Producing dummy variables for categorical data and cleaning data

# In[6]:


dummiesdf = pd.get_dummies(df['VisitorType'])
df.drop('VisitorType', inplace = True, axis = 1)
df['New_Visitor'] = dummiesdf['New_Visitor']
df['Other'] = dummiesdf['Other']
df['Returning_Visitor'] = dummiesdf['Returning_Visitor']


# In[7]:


dfmonth = pd.get_dummies(df['Month'])
df.drop('Month', inplace = True, axis = 1)
dfwithdummies = pd.concat([df, dfmonth], axis = 1, sort = False)


# In[8]:


dfwithdummies['Class'] = df['Revenue'].astype(int)
dfwithdummies.drop('Revenue', axis = 1, inplace = True)
dfwithdummies['Weekend'] = df['Weekend'].astype(int)
dfwithdummies.drop('Returning_Visitor', axis = 1, inplace = True)
dfcleaned = dfwithdummies.copy()


# In[9]:


X = dfcleaned.drop('Class', axis = 1)
Y = dfcleaned['Class'].copy()


# In[10]:


X.head()


# In[11]:


X.info()


# ## Checking for Collinearity Between Features and Creating Reducing Feature Size

# In[12]:


cor = X.corr()


# In[13]:


sns.heatmap(cor, xticklabels=cor.columns,yticklabels=cor.columns)


# In[14]:


# dfcleaned[dfcleaned['New_Visitor'] == 1]['Administrative'].value_counts()
def AvgMinutes(Count, Duration):
    if Duration == 0:
        output = 0
    elif Duration != 0:
        output = float(Duration)/float(Count)
    return output

Columns = [['Administrative', 'Administrative_Duration'], ['Informational', 'Informational_Duration'], ['ProductRelated', 'ProductRelated_Duration']]


X['AvgAdministrative'] = X.apply(lambda x: AvgMinutes(Count = x['Administrative'], Duration = x['Administrative_Duration']), axis = 1)
X['AvgInformational'] = X.apply(lambda x: AvgMinutes(Count = x['Informational'], Duration = x['Informational_Duration']), axis = 1)
X['AvgProductRelated'] = X.apply(lambda x: AvgMinutes(Count = x['ProductRelated'], Duration = x['ProductRelated_Duration']), axis = 1)
X.drop(['Administrative', 'Administrative_Duration','Informational', 'Informational_Duration','ProductRelated', 'ProductRelated_Duration'], axis = 1, inplace = True)


# In[ ]:


cor = X.corr()
sns.heatmap(cor, xticklabels=cor.columns,yticklabels=cor.columns)


# ## Quick overview of features

# In[15]:


# Histogram of all features
for idx,column in enumerate(X.columns):
    plt.figure(idx)
    X.hist(column=column,grid=False)


# In[ ]:


# Checking for NA values
for i in X.columns:
    print('Feature:',i)
    print('# of N/A:',X[i].isna().sum())


# In[16]:


for i in X_train.columns:
    print('####################')
    print('COLUMN TITLE:',i)
    print('# UNIQUE VALUES:',len(X_train[i].unique()))
    print('UNIQUE VALUES:',X_train[i].unique())
    print('####################')
    print()


# In[17]:


# Scaling to normalize data
X_copy = X.copy()
rc = RobustScaler()
X_rc=rc.fit_transform(X_copy)
X_rc=pd.DataFrame(X_rc,columns=X.columns)

for idx,column in enumerate(X_rc.columns):
    plt.figure(idx)
    X_rc.hist(column=column,grid=False)


# ## Linear Model with All Features

# In[18]:


from sklearn import linear_model
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X_rc,Y,test_size=.2)


# In[19]:


# Linear model
model = linear_model.SGDClassifier()
model.fit(X_train, y_train)


# In[20]:


y_pred = model.predict(X_test)


# In[21]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[22]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# ## Random Forest with all Features

# In[23]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=17, random_state=0)
clf.fit(X_train, y_train)
y_pred1 = clf.predict(X_test)
# In[24]:


accuracy_score(y_test, y_pred1)


# In[25]:


roc_auc_score(y_test, y_pred1)


# ## Finding Important Features then Removing from Dataframe

# In[26]:


from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
list_one =[]
feature_ranking = SelectKBest(chi2, k=5)
fit = feature_ranking.fit(X, Y)

fmt = '%-8s%-20s%s'

for i, (score, feature) in enumerate(zip(feature_ranking.scores_, X.columns)):
    list_one.append((score, feature))

dfObj = pd.DataFrame(list_one)
dfObj.sort_values(by=[0], ascending = False)


# In[27]:


X_rc.drop(['Aug','TrafficType','OperatingSystems','Other','Jul'],axis=1,inplace=True)


# In[28]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X_rc,Y,test_size=.2)


# ## Random Forest Classifier with Feature Selection Dataframe

# In[29]:


clf1 = RandomForestClassifier(n_estimators= 200, max_depth = 30 )
clf1.fit(X_train1, y_train1)
y_pred2 = clf1.predict(X_test1)


# In[30]:


accuracy_score(y_test1, y_pred2)


# In[31]:


roc_auc_score(y_test1, y_pred2)


# ## XGBoost Classifier with Feature Selection Dataframe

# In[32]:


model = XGBClassifier(learning_rate = 0.1, n_estimators=150, min_child_weight=3,  max_depth=13)
model.fit(X_train1, y_train1)


# In[33]:


y_pred3 = model.predict(X_test1)


# In[34]:


accuracy_score(y_test1, y_pred3)


# In[35]:


roc_auc_score(y_test1, y_pred3)


# # LogisticRegression with Feature Selection Dataframe

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
log_reg = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter = 10000)
log_reg.fit(X_train1,y_train1)
y_pred4 = log_reg.predict(X_test1)
print(accuracy_score(y_pred4,y_test1))
print(roc_auc_score(y_test1, y_pred4))


# ## Gaussian Naive Bayes with Feature Selection Dataframe

# In[45]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train1,y_train1)
y_pred5 = GNB.predict(X_test1)
print(accuracy_score(y_pred5,y_test1))
print(roc_auc_score(y_test1, y_pred5))


# # KNN classifier with Feature Selection Dataframe

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train1,y_train1)
y_pred6 = knn.predict(X_test1)
print(accuracy_score(y_pred6,y_test1))
print(roc_auc_score(y_test1, y_pred6))


# In[52]:


X_train1.shape


# # SVM Classification with PCA feature reduction technique

# In[58]:


from sklearn.decomposition import PCA
pca = PCA(n_components=15)
d=pca.fit_transform(X_train1)
e=pca.fit_transform(X_test1)
print(pca.explained_variance_ratio_.sum())


# In[59]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(d,y_train1)
y_pred7 = svm.predict(e)
print(accuracy_score(y_pred7,y_test1))
print(roc_auc_score(y_test1, y_pred7))


# # SVM Classification with  Feature Selection Dataframe

# In[60]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train1,y_train1)
y_pred8 = svm.predict(X_test1)
print(accuracy_score(y_pred8,y_test1))
print(roc_auc_score(y_test1, y_pred8))


# # Neural Network Classifier With Feature Selection Dataframe

# In[68]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(19,19,19), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train1,y_train1)
y_pred9= mlp.predict(X_test1)
print(accuracy_score(y_pred9,y_test1))
print(roc_auc_score(y_test1, y_pred9))


# In[ ]:
