#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import json
ratio=0.85


test_user_file="../data/data_source/test/test_userdata_180581.csv"
train_user_file="../data/data_source/train/train_category.csv"
submission_without_user="./to_submit/submission_without_user.json"
submission_with_user="./to_submit/submission_with_user.json"

merge_file="./to_submit/submission_merge_"+str(ratio)+".json"

df=pd.read_csv(test_user_file)
df_train=pd.read_csv(train_user_file)
df["post_id"]="post"+df["pid"].astype(str)
df


# In[3]:


with open(submission_without_user) as f:
    result_without_user=json.load(f)
with open(submission_with_user) as f:
    result_with_user=json.load(f)
df_without_user=pd.DataFrame(result_without_user["result"])
df_with_user=pd.DataFrame(result_with_user["result"])
df_with_user=df_with_user.rename(columns={"popularity_score":"with_user_score"})
df_without_user=df_without_user.rename(columns={"popularity_score":"without_user_score"})


# # 有爬虫的

# In[4]:


df1=df.dropna(axis=0,how="any",inplace=False)


# In[5]:


df11=pd.merge(pd.merge(df1,df_with_user),df_without_user)
df11


# In[6]:


plt.hist(df11["without_user_score"]-df11["with_user_score"],bins=50)


# In[7]:


df11["popularity_score"]=df11["without_user_score"]*0.02+df11["with_user_score"]*0.98
df11




# # 没爬虫的

# In[8]:


df2=df[df["meanviews"].isnull()]
df2


# In[10]:


df_with_user=df_with_user.rename(columns={"popularity_score":"with_user_score"})
df_with_user


# In[9]:


df_without_user=df_without_user.rename(columns={"popularity_score":"without_user_score"})
df_without_user


# In[11]:


df2=pd.merge(pd.merge(df2,df_with_user),df_without_user)
df2


# ## train和test共同的uid，且都没有爬虫数据

# In[12]:


common_uid=set(df_train["uid"])&set(df2["uid"])
df_common=pd.DataFrame({"uid":list(common_uid)})
df_common


# In[13]:


df22=pd.merge(df2[["pid","uid","follower","post_id","with_user_score","without_user_score"]],df_common)
df22


# In[14]:


plt.hist(df22["without_user_score"]-df22["with_user_score"],bins=100)


# In[15]:


df22["popularity_score"]=df22["without_user_score"]*0.95+df22["with_user_score"]*0.05
df22


# ## test的uid不在train中出现，且没有爬虫数据

# In[16]:


diff_uid=set(df2["uid"])-set(common_uid)
df_diff=pd.DataFrame({"uid":list(diff_uid)})
df_diff


# In[17]:


df33=pd.merge(df2[["pid","uid","follower","post_id","with_user_score","without_user_score"]],df_diff)
df33


# In[18]:


df33["popularity_score"]=(df33["without_user_score"]*ratio+df33["with_user_score"]*(1-ratio))
df33



# In[19]:


plt.hist(df33["without_user_score"]-df33["with_user_score"],bins=100)


# In[20]:


(df33["without_user_score"]-df33["with_user_score"]).mean(),(df33["without_user_score"]-df33["with_user_score"]).std()


# In[21]:


# df33["popularity_score"]=df33["without_user_score"]-1.064
# df33



# # result

# In[22]:


df_merge=pd.concat([df11,df22,df33],axis=0)
df_merge


# In[23]:


df_merge=pd.merge(df,df_merge)
df_merge=df_merge[["post_id","popularity_score"]]
df_merge


# In[24]:


result= [
        {
            "post_id": df_merge["post_id"][i],
            "popularity_score":round(float(df_merge["popularity_score"][i]), 4)
        } for i in range(len(df_merge))
]
# result


# In[25]:


submission = {
        "version": "VERSION 0.1",
        "result": result,
        "external_data": {
            "used": "true",
            "details": "ResNext pre-trained on ImageNet training set"
        }
    }


# In[26]:


filepath =merge_file
with open(filepath, "w") as file:
    json.dump(submission, file)




