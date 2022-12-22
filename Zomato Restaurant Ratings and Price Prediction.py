#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.offline as py 
import plotly.express as px
import plotly.graph_objs as go 
import matplotlib.ticker as ntick


# In[2]:


print(plt.style.available)


# In[43]:


plt.style.use("Solarize_Light2")


# In[4]:


from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[5]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


zomato_data = pd.read_csv ("//Zomato Restaurant Ratings and Price Prediction using Machine Learning//zomato.csv")


# In[12]:


zomato_data.shape


# In[13]:


zomato_data.head()


# In[14]:


zomato_data.tail()


# In[15]:


zomato_data.dtypes


# In[17]:


zomato_data.duplicated().sum()


# In[18]:


zomato_data.isnull().sum()


# In[19]:


zomato_data.describe()


# In[23]:


zomato_data.nunique()


# In[25]:


zomato_data.drop(columns=['url' , 'phone'] , inplace=True)


# In[26]:


zomato_data.columns


# In[27]:


zomato_data.duplicated().sum()


# In[28]:


zomato_data.drop_duplicates(inplace=True)


# In[29]:


zomato_data.duplicated().sum()


# In[30]:


zomato_data = zomato_data.dropna(axis = 0, how ='any')


# In[31]:


zomato_data.isnull().sum()


# In[32]:


zomato_data.shape


# In[33]:


zomato_data.rename(columns = {'approx_cost(for two people)':'approx_cost' ,'listed_in(type)':'type' ,'listed_in(city)':'city' }, inplace = True)


# In[34]:


zomato_data.columns


# In[35]:


zomato_data['online_order'].unique()


# In[55]:


plt.figure(figsize=(10,8))
color=['red','blue']
sns.countplot(x='online_order', data =zomato_data , palette = color)
plt.xticks(rotation =0)
plt.show()


# In[54]:


zomato_data['online_order'].value_counts()


# In[60]:


zomato_data['approx_cost'] = zomato_data['approx_cost'].apply(lambda x: x.replace(',','')) 
zomato_data['approx_cost'] = zomato_data['approx_cost'].astype(float)                                                     


# In[61]:


zomato_data.dtypes


# In[62]:


zomato_data['rate'].unique()


# In[63]:


zomato_data = zomato_data.loc[zomato_data.rate !='NEW']


# In[64]:


zomato_data['rate'].unique()


# In[89]:


zomato_data['rate'] = zomato_data['rate'].apply(lambda x: x.replace('/5','')) 
zomato_data['rate'] = zomato_data['rate'].apply(lambda x: x.replace(',',''))
zomato_data['rate'] = zomato_data['rate'].astype(float) 


# In[90]:


zomato_data['rate'].unique()


# In[69]:


top_20_names=zomato_data['name'].value_counts().head(20)


# In[72]:


plt.figure(figsize=(17,10))
data = zomato_data['name'].value_counts().head(20)
sns.barplot(x=data,y=data.index,palette='hls')
plt.title("Most famous restaurants chains in Bangaluru")
plt.xlabel("Number of outlets")
plt.show()


# In[74]:


zomato_data['book_table'].unique()


# In[75]:


zomato_data['book_table'].value_counts()


# In[78]:


palette_color = sns.color_palette('bright')
data=zomato_data['book_table'].value_counts()
plt.pie(data, labels=data.index, colors=palette_color, autopct='%.0f%%')
plt.show()


# In[91]:


zomato_data.dtypes


# In[92]:


plt.figure(figsize=(10,9))
sns.distplot(zomato_data['rate'] , bins =30)
plt.show()


# In[93]:


zomato_data.info()


# In[94]:


plt.figure(figsize=(10,9))
sns.distplot(zomato_data['votes'] , bins =30)
plt.show()


# In[104]:


plt.figure(figsize=(10,9))
sns.set_style("darkgrid")
sns.scatterplot(data=zomato_data, x="rate", y="approx_cost", hue="online_order" )
plt.show()


# In[105]:


zomato_data['type'].value_counts()


# In[107]:


plt.figure(figsize=(14,10))
sns.countplot(zomato_data['type'],data=zomato_data ,palette='hls')
plt.show()


# In[108]:


zomato_data['dish_liked'][0]


# In[109]:


zomato_data.shape[0]


# In[110]:


zomato_data.shape[1]


# In[116]:


import re

zomato_data.index=range(zomato_data.shape[0])
likes=[]
for i in range(zomato_data.shape[0]):
    array_split=re.split(',',zomato_data['dish_liked'][i])
    for item in array_split:
        likes.append(item)
            

   


# In[118]:


fav_food= pd.Series(likes).value_counts()
print(fav_food.head(20))


# In[121]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
zomato_data['online_order'].unique()
zomato_data['online_order']= label_encoder.fit_transform(zomato_data['online_order'])
zomato_data['online_order'].value_counts()


# In[124]:


zomato_data.info()


# In[123]:


zomato_data['book_table'].unique()
zomato_data['book_table']= label_encoder.fit_transform(zomato_data['book_table'])
zomato_data['book_table'].value_counts()


# In[125]:


zomato_data['location']= label_encoder.fit_transform(zomato_data['location'])
zomato_data['rest_type']= label_encoder.fit_transform(zomato_data['rest_type'])
zomato_data['cuisines']= label_encoder.fit_transform(zomato_data['cuisines'])
zomato_data['menu_item']= label_encoder.fit_transform(zomato_data['menu_item'])


# In[126]:


zomato_data.info()


# In[127]:


zomato_data.head()


# In[128]:


zomato_data.drop(columns=['address','name','dish_liked','reviews_list','type'],inplace=True)


# In[129]:


zomato_data.head()


# In[130]:


zomato_data.drop(columns=['city'] , inplace=True)


# In[131]:


zomato_data.head()


# In[133]:


X = zomato_data.iloc[: , [0,1,3,4,5,6,7,8]]


# In[134]:


X.head()


# In[135]:


Y= zomato_data['rate']


# In[136]:


Y.head()


# In[137]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)


# In[138]:


linear_regression_model=LinearRegression()
linear_regression_model.fit(x_train,y_train)


# In[139]:


from sklearn.metrics import r2_score
y_pred=linear_regression_model.predict(x_test)
r2_score(y_test,y_pred)


# In[140]:


extratrees_model=ExtraTreesRegressor(n_estimators = 140)
extratrees_model.fit(x_train,y_train)
y_predict=extratrees_model.predict(x_test)
r2_score(y_test,y_predict)


# In[ ]:




