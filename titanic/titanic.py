#!/usr/bin/env python
# coding: utf-8

# # load dataset
# just read the csv..

# In[9]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[10]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train


# In[11]:


# split train data to features and labels
train_x = train.drop(["Survived"], axis=1)
train_y = train["Survived"]

# copy test as test_x (no survive in test!)
test_x = test.copy()

train_x


# # cleanup data
# - 乗客ID,名前,TIcket,Cabinなどは良い特徴量ではないため、取り除いたほうが良い。
# - またgradient boosting decision treeには文字列は入力できない
# →SexとEmbarkedを数字に変換する。
# 

# In[12]:


# drop id, name, ticket, cabin information
train_x = train_x.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
train_x


# In[13]:


# convert sex and embarked to int
# LabelEncoderを使うことで自動的にデータを変換してくれる。ありがたい。


for c in ["Sex", "Embarked"]:
    # how to process NaN
    le = LabelEncoder()
    le.fit(train_x[c].fillna("NA"))
    
    # transform train and test data
    train_x[c] = le.transform(train_x[c].fillna("NA"))
    test_x[c] = le.transform(test_x[c].fillna("NA"))


# In[14]:


# monitor transformed data
train_x


# # build model

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


# build model
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# predict test data
pred = model.predict(test_x)[:, 1]
print(model.predict(test_x))

# binarize prediction
pred_label = np.where(pred > 0.5, 1, 0)
pred_label


# 
# # get final results for upload

# In[ ]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred_label})
submission.to_csv("submission.csv", index=False)


# that's all :)

# In[ ]:




