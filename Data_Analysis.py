#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

import xgboost as xgb
import optuna
from optuna import Trial, visualization


# In[112]:


def EDA(data):
    print(data.info())
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    print(data.describe())

    df_not_num = data[['Joint1 type', 'Joint1 axis', 'Joint2 type', 'Joint2 axis',
       'Joint3 type', 'Joint3 axis',
       'Joint4 type', 'Joint4 axis',  'Joint5 type',
       'Joint5 axis', 'Joint6 type', 'Joint6 axis',]]
    df_num = data[["Link1 length","Link2 length","Link3 length","Link4 length","Link5 length","Link6 length",'Success_Rates', 'Min_Manipulability', 'Max_Manipulability', 'Manipulability_Rates']]

# Box-Plot
    y1 = list(data['Success_Rates'])
    plt.boxplot(y1)
    plt.title("Box-Plot - Success_Rates ")
    plt.show()

    y2 = list(data['Manipulability_Rates'])
    plt.boxplot(y2)
    plt.title("Box-Plot - Manipulability_Rates ")
    plt.show()

# Histogram
    plt.figure(figsize=(9, 8))
    sns.histplot(data['Success_Rates'], color='g')
    plt.title("Histogram - Success_Rates")
    plt.show()

    plt.figure(figsize=(9, 8))
    sns.histplot(data['Manipulability_Rates'], color='g')
    plt.title("Histogram - Manipulability_Rates")
    plt.show()

# Categorical plots
    fig, axes = plt.subplots(round(len(df_not_num.columns) / 3), 3, figsize=(7, 35))

    for i, ax in enumerate(fig.axes):
        if i < len(df_not_num.columns):
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
            sns.countplot(x=df_not_num.columns[i], alpha=0.7, data=df_not_num, ax=ax)

    fig.tight_layout()
    plt.show()

# Correlation
    corr = df_num.corr()  # We already examined SalePrice correlations
    plt.figure(figsize=(12, 10))

    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],
                cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                annot=True, annot_kws={"size": 8}, square=True)
    plt.show()

    print("Counting unique values of categorical variables \n",df_not_num.nunique())
    print("Counting unique values of arm's configuration \n", data[["Link1 length","Link2 length","Link3 length","Link4 length","Link5 length","Link6 length"]].nunique())
    print("Describing numeric variables \n", df_num.describe())


# In[113]:


''' read data and insert into a combined dataframe'''

data1 = pd.read_csv('sim_resultsNUC1.csv')
data2 = pd.read_csv('sim_resultsNUC2.csv')
data22 = pd.read_csv('sim_resultsNUC3.csv')

data3 = pd.read_csv('sim_resultslinux1.csv')
data4 = pd.read_csv('sim_resultslinux2.csv')
data5 = pd.read_csv('sim_resultslinux3.csv')

to_merge = [data1,data2,data22,data3,data4,data5]
data = pd.concat(to_merge, ignore_index=True)
data = data.drop(columns=['Unnamed: 0','Unnamed: 9'])
print("Dataframe Columns",data.columns)


# In[114]:


data.shape


# In[115]:


'''Data Understanding'''

''' OPTION 1 - PREDICT MEASUREMENTS BY ARM'S CONFIGURATION '''
''' Merge the data of the 10 points of each robotic arm to 1 sample'''
grouped_data = data.groupby(['Arm ID']).agg(Success_Rates=pd.NamedAgg (column="Sucsses", aggfunc= 'mean'),
                                                   Min_Manipulability =pd.NamedAgg (column= 'Manipulability - roni',aggfunc ='min'),
                                                Max_Manipulability = pd.NamedAgg (column= 'Manipulability - roni',aggfunc ='max'),
                                            Manipulability_Rates = pd.NamedAgg (column= 'Manipulability - roni',aggfunc ='mean')).reset_index()
with pd.option_context('display.max_rows', 5,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print("Data OPTION 1 - PREDICT MEASUREMENTS BY ARM'S CONFIGURATION: .\n ", grouped_data)


# In[116]:


''' count negative values in Manipulability column'''
subset = grouped_data[(grouped_data["Min_Manipulability"] < 0) & (grouped_data["Max_Manipulability"] < 0)]
print("negative values in Manipulability columns\n",subset.count())
subset


# In[117]:


''' Drop all arms with manipulability (-1) - Min and Max'''
grouped_data = grouped_data.drop(grouped_data[(grouped_data["Min_Manipulability"] < 0) & (grouped_data["Max_Manipulability"] < 0)].index)
grouped_data = grouped_data.reset_index(drop=True)
print(grouped_data)


# In[118]:


''' count negative values in Manipulability column'''
subset = grouped_data[(grouped_data["Manipulability_Rates"] < 0)]
print("negative values in Manipulability columns\n",subset.count())
print(subset)


# In[119]:


''' Drop all arms with negative manipulability rate'''
grouped_data = grouped_data.drop(grouped_data[(grouped_data["Manipulability_Rates"] < 0)].index)
grouped_data = grouped_data.reset_index(drop=True)
print(grouped_data)


# In[120]:


''' count NULL values by column'''
print("NULL values by column\n",grouped_data.isnull().sum(axis = 0))


''' Analyze Success Rates'''
max_success = grouped_data["Success_Rates"].max()
min_success = grouped_data["Success_Rates"].min()

print("\nMax success", max_success)
print("Min success", min_success)

''' Analyze Manipulability Rates'''
max_mani_rate = grouped_data["Manipulability_Rates"].max()
min_mani_rate = grouped_data["Manipulability_Rates"].min()

print("\nMax Manipulability_Rates", max_mani_rate)
print("Min Manipulability_Rates", min_mani_rate)


# In[121]:


'''Drop all rows with NULL Manipulability'''
grouped_data = grouped_data.dropna()
grouped_data = grouped_data.reset_index(drop=True)
print(grouped_data)


# In[122]:


'''Check for duplicate arms '''
df_duplicates = grouped_data.duplicated(subset = ['Arm ID'])
df_duplicates.value_counts()


# In[123]:


''' split arm name to features '''
# new data frame with split value columns
new = grouped_data["Arm ID"].str.split("_", expand=True)
print("Dataframe with splitting Arm's name: ",new)

grouped_data["Joint1 type"] = new[1]
grouped_data["Joint1 axis"] = new[2]
grouped_data["Link1 length"] = new[3]+"."+new[4]

grouped_data["Joint2 type"] = new[5]
grouped_data["Joint2 axis"] = new[6]
grouped_data["Link2 length"] = new[7]+"."+new[8]

grouped_data["Joint3 type"] = new[9]
grouped_data["Joint3 axis"] = new[10]
grouped_data["Link3 length"] = new[11]+"."+new[12]

grouped_data["Joint4 type"] = new[13]
grouped_data["Joint4 axis"] = new[14]
grouped_data["Link4 length"] = new[15]+"."+new[16]

grouped_data["Joint5 type"] = new[17]
grouped_data["Joint5 axis"] = new[18]
grouped_data["Link5 length"] = new[19]+"."+new[20]

grouped_data["Joint6 type"] = new[21]
grouped_data["Joint6 axis"] = new[22]
grouped_data["Link6 length"] = new[23]+"."+new[24]

grouped_data = grouped_data.drop(['Arm ID'], axis=1)
print("Columns Order", grouped_data.columns)


# In[124]:


grouped_data = grouped_data[['Joint1 type', 'Joint1 axis', 'Link1 length', 'Joint2 type', 'Joint2 axis',
       'Link2 length', 'Joint3 type', 'Joint3 axis', 'Link3 length',
       'Joint4 type', 'Joint4 axis', 'Link4 length', 'Joint5 type',
       'Joint5 axis', 'Link5 length', 'Joint6 type', 'Joint6 axis',
       'Link6 length','Success_Rates', 'Min_Manipulability', 'Max_Manipulability', 'Manipulability_Rates']]

with pd.option_context('display.max_rows', 5,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print("Data after features organizing: ",grouped_data)


# In[125]:


grouped_data[['Success_Rates', 'Min_Manipulability', 'Max_Manipulability', 'Manipulability_Rates']] = grouped_data[['Success_Rates', 'Min_Manipulability', 'Max_Manipulability', 'Manipulability_Rates']].astype('float64')
grouped_data[["Joint1 type","Joint2 type","Joint3 type","Joint4 type","Joint5 type","Joint6 type"]] = grouped_data[["Joint1 type","Joint2 type","Joint3 type","Joint4 type","Joint5 type","Joint6 type"]].astype("string")
grouped_data[["Joint1 axis","Joint2 axis","Joint3 axis","Joint4 axis","Joint5 axis","Joint6 axis"]]= grouped_data[["Joint1 axis","Joint2 axis","Joint3 axis","Joint4 axis","Joint5 axis","Joint6 axis"]] .astype("string")
grouped_data[["Link1 length","Link2 length","Link3 length","Link4 length","Link5 length","Link6 length"]] = grouped_data[["Link1 length","Link2 length","Link3 length","Link4 length","Link5 length","Link6 length"]] .astype('float64')


# In[126]:


grouped_data.iloc[[12484]]


# In[58]:


''' count NULL values by column'''
grouped_data = grouped_data.drop(index= 12484 )
grouped_data = grouped_data.reset_index(drop=True)


# In[59]:


''' Data Pre-Processing/ Data Preparation '''
'''Drop Joint1 type,Joint1 axis,Link1 length'''
grouped_data = grouped_data.drop(columns=["Joint1 type","Joint1 axis","Link1 length"])
grouped_data = grouped_data.drop(columns=['Manipulability_Rates', 'Max_Manipulability'])
with pd.option_context('display.max_rows', 5,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print("Data after dropping irrelevant columns: ",grouped_data)


# In[60]:


grouped_data.columns


# In[98]:


''' Analysis joint's properties v.s performance indicator '''

fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(221)
g = sns.stripplot(ax = ax1 ,x='Joint6 axis', y="Min_Manipulability", data=grouped_data) # pass ax1

ax2 = fig.add_subplot(222)
g = sns.stripplot(ax = ax2, x='Joint6 type', y="Min_Manipulability", data=grouped_data) # pass ax2

ax3 = fig.add_subplot(223)
g = sns.boxplot(ax = ax3, x='Joint6 type', y="Success_Rates", data=grouped_data) # pass ax3

ax4 = fig.add_subplot(224)
g = sns.boxplot(ax = ax4, x='Joint6 axis', y="Success_Rates", data=grouped_data) # pass ax4

plt.tight_layout()


# In[99]:


fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(221)
g = sns.stripplot(ax = ax1 ,x='Joint5 axis', y="Min_Manipulability", data=grouped_data) # pass ax1

ax2 = fig.add_subplot(222)
g = sns.stripplot(ax = ax2, x='Joint5 type', y="Min_Manipulability", data=grouped_data) # pass ax2

ax3 = fig.add_subplot(223)
g = sns.boxplot(ax = ax3, x='Joint5 type', y="Success_Rates", data=grouped_data) # pass ax3

ax4 = fig.add_subplot(224)
g = sns.boxplot(ax = ax4, x='Joint5 axis', y="Success_Rates", data=grouped_data) # pass ax4

plt.tight_layout()


# In[100]:


fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(221)
g = sns.stripplot(ax = ax1 ,x='Joint4 axis', y="Min_Manipulability", data=grouped_data) # pass ax1

ax2 = fig.add_subplot(222)
g = sns.stripplot(ax = ax2, x='Joint4 type', y="Min_Manipulability", data=grouped_data) # pass ax2

ax3 = fig.add_subplot(223)
g = sns.boxplot(ax = ax3, x='Joint4 type', y="Success_Rates", data=grouped_data) # pass ax3

ax4 = fig.add_subplot(224)
g = sns.boxplot(ax = ax4, x='Joint4 axis', y="Success_Rates", data=grouped_data) # pass ax4

plt.tight_layout()


# In[101]:


fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(221)
g = sns.stripplot(ax = ax1 ,x='Joint3 axis', y="Min_Manipulability", data=grouped_data) # pass ax1

ax2 = fig.add_subplot(222)
g = sns.stripplot(ax = ax2, x='Joint3 type', y="Min_Manipulability", data=grouped_data) # pass ax2

ax3 = fig.add_subplot(223)
g = sns.boxplot(ax = ax3, x='Joint3 type', y="Success_Rates", data=grouped_data) # pass ax3

ax4 = fig.add_subplot(224)
g = sns.boxplot(ax = ax4, x='Joint3 axis', y="Success_Rates", data=grouped_data) # pass ax4

plt.tight_layout()


# In[102]:


fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(221)
g = sns.stripplot(ax = ax1 ,x='Joint2 axis', y="Min_Manipulability", data=grouped_data) # pass ax1

ax2 = fig.add_subplot(222)
g = sns.stripplot(ax = ax2, x='Joint2 type', y="Min_Manipulability", data=grouped_data) # pass ax2

ax3 = fig.add_subplot(223)
g = sns.boxplot(ax = ax3, x='Joint2 type', y="Success_Rates", data=grouped_data) # pass ax3

ax4 = fig.add_subplot(224)
g = sns.boxplot(ax = ax4, x='Joint2 axis', y="Success_Rates", data=grouped_data) # pass ax4

plt.tight_layout()


# In[103]:


'''Data analysis- By arm'''
'''Most successful in terms of reachability'''
grouped_data[grouped_data['Success_Rates']==grouped_data['Success_Rates'].max()]


# In[104]:


'''Most successful in terms of manipulability'''
grouped_data[grouped_data['Min_Manipulability']==grouped_data['Min_Manipulability'].max()]


# In[106]:


'''Best arms for both indicators'''
best_of_two = grouped_data[((grouped_data['Min_Manipulability'] >= 2) & (grouped_data['Success_Rates'] > 0.6))]
best_of_two


# In[134]:


plt.scatter(best_of_two['Success_Rates'],best_of_two['Min_Manipulability'], c='coral')
plt.title('Success Rates V.S Manipulability for the sucssesful arms')
plt.xlabel('Success Rates')
plt.ylabel('Manipulability')


# In[143]:


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(1, 2)
sns.countplot(x="Joint6 type", data=best_of_two,ax=axes[0] )
sns.countplot(x="Joint6 axis", data=best_of_two,ax=axes[1])


# In[144]:


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(1, 2)
sns.countplot(x="Joint5 type", data=best_of_two,ax=axes[0] )
sns.countplot(x="Joint5 axis", data=best_of_two,ax=axes[1])


# In[145]:


'''Non successful in terms of manipulability'''
worst_of_two = grouped_data[((grouped_data['Min_Manipulability'] < 2) & (grouped_data['Success_Rates'] <= 0.6))]
worst_of_two


# In[147]:


plt.scatter(worst_of_two['Success_Rates'],worst_of_two['Min_Manipulability'], c='coral')
plt.title('Success Rates V.S Manipulability for the non sucssesful arms')
plt.xlabel('Success Rates')
plt.ylabel('Manipulability')


# In[148]:


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(1, 2)
sns.countplot(x="Joint6 type", data=worst_of_two,ax=axes[0] )
sns.countplot(x="Joint6 axis", data=worst_of_two,ax=axes[1])


# In[150]:


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(1, 2)
sns.countplot(x="Joint5 type", data=worst_of_two,ax=axes[0] )
sns.countplot(x="Joint5 axis", data=worst_of_two,ax=axes[1])


# In[133]:


plt.scatter(grouped_data['Success_Rates'],grouped_data['Min_Manipulability'], c='lightblue')
plt.title('Success Rates V.S Manipulability')
plt.xlabel('Success Rates')
plt.ylabel('Manipulability')


# In[21]:


'''Get Dummies'''
grouped_data = pd.get_dummies(grouped_data, columns = ["Joint2 type","Joint3 type","Joint4 type","Joint5 type","Joint6 type","Joint2 axis","Joint3 axis","Joint4 axis","Joint5 axis","Joint6 axis"])
with pd.option_context('display.max_rows', 5,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print("Data after dummies: ",grouped_data)


# In[22]:


'cronbach alpha'
import pingouin as pg
pg.cronbach_alpha(data=grouped_data[['Min_Manipulability','Success_Rates']])


# In[25]:


grouped_data.describe()


# In[570]:


''' Modeling '''
''' split to train, test and validation sets '''
n_targets = 2

train_df = grouped_data.sample(frac=0.8,random_state=200) #random state is a seed value
test_df = grouped_data.drop(train_df.index)

X = train_df.drop(columns=['Manipulability_Rates','Success_Rates']).values
Y = train_df[['Manipulability_Rates','Success_Rates']].values
X_train, x_valid, y_train, y_valid = train_test_split(train_df.drop(columns=['Manipulability_Rates','Success_Rates'] ), train_df[['Manipulability_Rates','Success_Rates']], test_size=0.3)
x_test = test_df.drop(columns=['Manipulability_Rates','Success_Rates'] )
y_test = test_df[['Manipulability_Rates','Success_Rates']]

# y_train = (np.tile(y_train, (n_targets,1)))
# y_valid = (np.tile(y_valid, (n_targets,1)))
# y_test = (np.tile(y_test, (n_targets,1)))

# x_train = np.delete(x_train, 3753,0)
# y_train = np.delete(y_train, 3753,0)

x_train =X_train.to_numpy()
y_train=y_train.to_numpy()
x_valid =x_valid.to_numpy()
y_valid =y_valid.to_numpy()
x_test =x_test.to_numpy()
y_test =y_test.to_numpy()


print("x train\n", x_train.shape)
print("y train\n", y_train.shape)

print("x valid\n", x_valid.shape)
print("y valid\n", y_valid.shape)

print("x test\n", x_test.shape)
print("y test\n", y_test.shape)


# In[631]:


'''Covariance ot the two outputs - indicates the relationship of two variables'''
grouped_data[['Manipulability_Rates', 'Success_Rates']].cov()


# In[ ]:


'''Variance ot the two outputs - Manipulability_Rates is normalized'''
grouped_data[['Manipulability_Rates', 'Success_Rates']].std()


# In[152]:


''' OPTION 2 - group by point '''
''' Merge the data of the 10 points of each POINT'''
grouped_points_data = data.groupby(['Point number']).agg(Success_Rates=pd.NamedAgg (column="Sucsses", aggfunc= 'mean'),
                                                   Min_Manipulability =pd.NamedAgg (column= 'Manipulability - roni',aggfunc ='min'),
                                                Max_Manipulability = pd.NamedAgg (column= 'Manipulability - roni',aggfunc ='max'),
                                            Manipulability_Rates = pd.NamedAgg (column= 'Manipulability - roni',aggfunc ='mean')).reset_index()
with pd.option_context('display.max_rows', 10,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print("Data OPTION 2 - GROUP BY POINT: .\n ", grouped_points_data)


# In[ ]:


grouped_points_data[['Success_Rates', 'Min_Manipulability', 'Max_Manipulability', 'Manipulability_Rates']] = grouped_points_data[['Success_Rates', 'Min_Manipulability', 'Max_Manipulability', 'Manipulability_Rates']].astype('float64')
grouped_points_data[["Joint1 type","Joint2 type","Joint3 type","Joint4 type","Joint5 type","Joint6 type"]] = grouped_points_data[["Joint1 type","Joint2 type","Joint3 type","Joint4 type","Joint5 type","Joint6 type"]].astype("string")
grouped_points_data[["Joint1 axis","Joint2 axis","Joint3 axis","Joint4 axis","Joint5 axis","Joint6 axis"]]= grouped_points_data[["Joint1 axis","Joint2 axis","Joint3 axis","Joint4 axis","Joint5 axis","Joint6 axis"]] .astype("string")
grouped_points_data[["Link1 length","Link2 length","Link3 length","Link4 length","Link5 length","Link6 length"]] = grouped_points_data[["Link1 length","Link2 length","Link3 length","Link4 length","Link5 length","Link6 length"]] .astype('float64')

