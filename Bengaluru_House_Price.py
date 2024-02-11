#!/usr/bin/env python
# coding: utf-8

# # Data Loading

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv(r"C:\Users\Pandilla Lakshmaiah\Desktop\Project_dataset\Bengaluru_House_Data.csv")
data


# In[3]:


data.info()


# In[4]:


data.columns


# # Data Cleaning

# In[5]:


data=data.drop(['area_type','availability','balcony','society'],axis=1)
data


# In[6]:


data.isna().sum()


# In[7]:


data=data.dropna()


# In[8]:


data.isna().sum()


# In[9]:


data.shape


# In[10]:


data['size'].unique()


# 
# # Feature Engineering
# **Add new feature(integer) for bhk (Bedrooms Hall Kitchen)**

# In[11]:


data['BHK']=data['size'].apply(lambda x: int(x.split(' ')[0]))


# In[12]:


data.head()


# In[13]:


data['BHK'].unique()


# In[14]:


data[data.BHK>20]


# In[15]:


data.total_sqft.unique()


# In[16]:


def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


# In[17]:


data[~data['total_sqft'].apply(isfloat)].head(10)


# In[18]:


def convert_sqft_tonum(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[19]:


data=data.copy()
data['total_sqft']=data['total_sqft'].apply(convert_sqft_tonum)


# In[20]:


data.head(10)


# In[21]:


data.loc[30]


# # Feature Engineering
# **Add new feature called price per square feet**

# In[22]:


data1=data.copy()
data1['price_per_sqft']=data1['price']*1000000/data1['total_sqft']
data1.head()


# In[23]:


len(data1.location.unique())


# In[24]:


data1.location=data1.location.apply(lambda x: x.strip())
location_stats=data1.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[25]:


len(location_stats[location_stats<=10])


# In[26]:


locationlessthan10=location_stats[location_stats<=10]
locationlessthan10


# In[27]:


len(data1.location.unique())


# In[28]:


data1.location=data1.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
len(data1.location.unique())


# In[29]:


data1.head(10)


# In[30]:


data1[data1.total_sqft/data1.BHK<300].head()


# In[31]:


data2=data1[~(data1.total_sqft/data1.BHK<300)]
data2.head(10)


# In[32]:


data2.shape


# # Outlier Removal Using Standard Deviation and Mean

# In[33]:


data2["price_per_sqft"].describe().apply(lambda x:format(x,'f'))


# In[34]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
data3=remove_pps_outliers(data2)
data3.shape


# In[35]:


import matplotlib.pyplot as plt
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.BHK==2)]
    bhk3=df[(df.location==location)&(df.BHK==3)]
    plt.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='Blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='green',marker='+',label='3 BHK',s=50)
    plt.xlabel('Total Square Foot')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
plot_scatter_chart(data3,"Rajaji Nagar")


# In[36]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats={}
        for BHK,BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'std':np.std(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats=bhk_sats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

data4=remove_bhk_outliers(data3)
data4.shape


# In[37]:


plot_scatter_chart(data4,"Rajaji Nagar")


# In[38]:


plt.rcParams['figure.figsize']=(20,15)
plt.hist(data4.price_per_sqft,rwidth=0.6)
plt.xlabel("Price Per Square Foor")
plt.ylabel("Count")


# 
# # Outlier Removal Using Bathrooms Feature

# In[39]:


data4.bath.unique()


# In[40]:


data4[data4.bath>10]


# In[41]:


plt.rcParams['figure.figsize']=(20,15)
plt.hist(data4.bath,rwidth=0.6)
plt.xlabel("Number Of Bathroom")
plt.ylabel("Count")


# In[42]:


data4[data4.bath>data4.BHK+2]


# In[43]:


data5=data4[data4.bath<data4.BHK+2]
data5.shape


# In[44]:


data6=data5.drop(['size','price_per_sqft'],axis='columns')
data6


# In[45]:


dummies=pd.get_dummies(data6.location)
dummies.head(10)


# In[46]:


data7=pd.concat([data6,dummies.drop('other',axis='columns')],axis='columns')
data7.head()


# In[47]:


data8=data7.drop('location',axis='columns')
data8.head()


# 
# # Build a Model Now...

# In[48]:


data8.shape


# In[49]:


X=data8.drop('price',axis='columns')
X.head()


# In[50]:


y=data8.price


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[52]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# # Use K Fold cross validation to measure accuracy of our LinearRegression model

# In[53]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# 
# # Find best model using GridSearchCV
# 

# In[54]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
import pandas as pd

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'n_jobs': [None, 1, 2, 3],
                'positive': [False, True]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Assuming X and y are defined somewhere above
result_df = find_best_model_using_gridsearchcv(X, y)
print(result_df)


# 
# **Based on above results we can say that LinearRegression gives the best score. Hence we will use that.**

# 
# # Test the model for few properties

# In[55]:


def price_predict(location,sqft,bath,BHK):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=BHK
    if loc_index >=0:
        x[loc_index]=1
    return model.predict([x])[0]


# In[56]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[57]:


price_predict('1st Phase JP Nagar',1000,2,3)


# In[58]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[59]:


price_predict('Indira Nagar',1000,2,2)


# In[ ]:




