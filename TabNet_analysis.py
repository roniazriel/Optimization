#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


grouped_data = pd.read_csv('grouped_data.csv') 


# In[3]:


grouped_data.shape


# In[4]:


grouped_data = grouped_data.drop('Unnamed: 0',axis=1)


# In[5]:


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

y_test


# In[6]:


# '''Save and load model'''
# save state dict
#path = 'TabModel'
#saved_filename = regressor.save_model(path)
# define new model and load save parameters
regressor = TabNetRegressor()
regressor.load_model('TabModel.zip')

print(regressor)


# In[7]:


from sklearn.metrics import r2_score

regressor.fit(x_train,y_train,
    eval_set=[(x_train, y_train), (x_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=1000,
    eval_metric=['mse','mae'])

preds = regressor.predict(x_test) ### Predictions on train set

manipulability_preds = [row[0] for row in preds]
success_preds = [row[1] for row in preds]
manipulability_true = [row[0] for row in y_test]
success_true = [row[1] for row in y_test]
preds_true_df = pd.DataFrame({'MANIPULABILITY PREDS': manipulability_preds,'SUCSSES RATE PREDS': success_preds, 'MANIPULABILITY TRUE VALUE': manipulability_true,'SUCSSES RATE TRUE VALUE': success_true}, columns=['MANIPULABILITY PREDS','SUCSSES RATE PREDS' ,'MANIPULABILITY TRUE VALUE','SUCSSES RATE TRUE VALUE'])

test_acc1 = r2_score(manipulability_true,manipulability_preds)
test_acc2 = r2_score( success_true,success_preds)

test_mse = mean_squared_error(y_pred=preds, y_true=y_test)
test_mae = mean_absolute_error(y_pred=preds, y_true=y_test)

print(f"BEST VALID SCORE: {regressor.best_cost}")
print(f"MSE TEST SCORE: {test_mse}")
print(f"MAE TEST SCORE: {test_mae}")
print(f"R SQURE FOR MANIPULABILITY TEST SCORE: {test_acc1}")
print(f"R SQURE FOR SUCSSES RATE TEST SCORE: {test_acc2}")

preds_true_df.to_csv('C:/Users/azrie/PycharmProjects/pythonProject/DL/predictions.csv')


# In[8]:


'''Manipulability scores'''
yhat = np.array(manipulability_preds)
SS_Residual = sum((np.array(manipulability_true)-yhat)**2)       
SS_Total = sum((manipulability_true-np.mean(manipulability_true))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(manipulability_true)-x_test.shape[1]-1)
print ('r_squared for manipulability:',r_squared)
print('adjusted_r_squared for manipulability:', adjusted_r_squared)


# In[9]:


'''Succses index scores'''
yhat = np.array(success_preds)
SS_Residual = sum((np.array(success_true)-yhat)**2)       
SS_Total = sum((success_true-np.mean(success_true))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(success_true)-x_test.shape[1]-1)
print ('r_squared for Succses index:',r_squared)
print('adjusted_r_squared for Succses index:', adjusted_r_squared)


# In[10]:


''' Errors- mean and std '''
errors_manipulability = np.array(manipulability_preds) - np.array(manipulability_true)
errors_sucsses = np.array(success_preds) - np.array(success_true)

mean_errors_manipulability = errors_manipulability.mean()
mean_errors_sucsses = errors_sucsses.mean()

std_errors_manipulability = errors_manipulability.std()
std_errors_sucsses = errors_sucsses.std()

print (errors_manipulability, errors_sucsses)
print (mean_errors_manipulability, mean_errors_sucsses)
print (std_errors_manipulability, std_errors_sucsses)


# In[11]:


'''Residuals V.S Predictions'''
residuals = np.array(manipulability_true) - np.array(manipulability_preds)
plt.scatter(residuals,manipulability_preds)
plt.xlabel("Residuals")
plt.ylabel("True Manipulability")
plt.title("Manipulability- TRUE V.S PREDICTED")
plt.show()


# In[12]:


q25, q75 = np.percentile(residuals, [25, 75])
bin_width = 2 * (q75 - q25) * len(residuals) ** (-1/3)
print(bin_width)
bins = round((residuals.max() - residuals.min()) / bin_width)
print("Freedman–Diaconis number of bins:", bins)
plt.hist(residuals, bins=bins)
plt.title('Histogram of Manipulability Residuals')
plt.ylabel('Count')
plt.xlabel('residuals')
plt.show()

#i=900
bins=np.linspace(-0.6,0.6,134)
#residuals *=i
t,e= np.histogram(residuals, bins=bins,density=True)
plt.bar(e[:-1], (t/12484)*100, width=0.01)
# print(t,'t', e, 'e')
plt.title('Probability of Manipulability Residuals')
plt.ylabel('Probability')
plt.xlabel('residuals')
plt.show()


# In[47]:


'''change reability predictions to the closest digit'''
rounded_reachability = np.round(success_preds,1)
print(rounded_reachability)
print(success_preds)


# In[48]:


'''Residuals V.S Predictions'''
residuals = np.array(success_true) - np.array(rounded_reachability)
plt.scatter(residuals,rounded_reachability)
plt.xlabel("Residuals")
plt.ylabel("True Sucsses")
plt.title("Sucsses- TRUE V.S PREDICTED")
plt.show()


# In[49]:


q25, q75 = np.percentile(residuals, [25, 75])
bin_width = 2 * (q75 - q25) * len(residuals) ** (-1/3)
bins = round((residuals.max() - residuals.min()) / bin_width)
print("Freedman–Diaconis number of bins:", bins)
plt.hist(residuals, bins=bins)
plt.title('Histogram of Success Residuals')
plt.ylabel('Count')
plt.xlabel('residuals');


# In[54]:


# residuals_manip_related_true = (np.array(manipulability_true) - np.array(manipulability_preds))/np.array(manipulability_true)
residuals_sucsses_related_true = (np.array(rounded_reachability) - np.array(success_true))/np.array(success_true)
'''Manipulability'''
# q25, q75 = np.percentile(residuals_manip_related_true, [25, 75])
# bin_width = 2 * (q75 - q25) * len(residuals_manip_related_true) ** (-1/3)
# bins = round((residuals_manip_related_true.max() - residuals_manip_related_true.min()) / bin_width)
# plt.hist(residuals_manip_related_true, bins=bins)
# plt.title('Manipulability errors in relation to the true value')
# plt.ylabel('Count')
# plt.xlabel('residuals');

'''Sucsses'''
q25, q75 = np.percentile(residuals_sucsses_related_true, [25, 75])
bin_width = 2 * (q75 - q25) * len(residuals_sucsses_related_true) ** (-1/3)
bins = round((residuals_sucsses_related_true.max() - residuals_sucsses_related_true.min()) / bin_width)
plt.hist(residuals_sucsses_related_true, bins=bins)
plt.title('Sucsses errors in relation to the true value')
plt.ylabel('Count')
plt.xlabel('residuals');


# In[55]:


'''Percentage of errors that are less than 0.2'''
errors_sucsses = np.array(success_true) - np.array(rounded_reachability)
errors_manip = np.array(manipulability_true) - np.array(manipulability_preds)

errors_sucsses=pd.DataFrame(errors_sucsses)
errors_manip= pd.DataFrame(errors_manip)
errors_sucsses.columns= ['reachability errors']

errors_manip.columns= ['manipulability errors']
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None) 
print(errors_sucsses)
print(errors_manip)

print("reachability index -  percentage of errors that are smaller then 0.1 - error of 1 cluster")
print((errors_sucsses[abs(errors_sucsses['reachability errors']<=0.1)].count()/len(errors_sucsses))*100)
print("manipulability index -  percentage of errors that are smaller then 0.2")
print((errors_manip[abs(errors_manip['manipulability errors']<=0.2)].count()/len(errors_manip))*100)


# In[146]:


'''True value v.s predicted'''
sns.scatterplot(x=manipulability_preds, y=manipulability_true,palette="deep")
plt.xlabel("Predicted Manipulability")
plt.ylabel("True Manipulability")
plt.title("MANIPULABILITY- TRUE V.S PREDICTED")
plt.show()

sns.scatterplot(x=success_preds, y=success_true,palette="deep")
plt.xlabel("Predicted Sucsses")
plt.ylabel("True Sucsses")
plt.title("Sucsses- TRUE V.S PREDICTED")
plt.show()


# In[131]:


'''Covariance ot the two outputs - indicates the relationship of two variables'''
grouped_data[['Manipulability_Rates', 'Success_Rates']].cov()


# In[98]:


'''Variance ot the two outputs - Manipulability_Rates is normalized'''
grouped_data[['Manipulability_Rates', 'Success_Rates']].std()

