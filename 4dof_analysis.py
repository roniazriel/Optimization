import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Load data'''
# dataset = pd.read_csv('4dof_valid_results.csv')
# dataset = dataset.drop(columns='Unnamed: 0')
# dataset.rename(columns = {'Arm ID':'Arm_ID'}, inplace = True)
#
# grouped_data = dataset.groupby(['Arm_ID'])['Success'].sum().astype(float).reset_index()
#
# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', None,
#                        'display.precision', 3):
#     print(grouped_data)
# print(len(grouped_data))
#
# good_arms = grouped_data.loc[grouped_data['Success'] > 9]
# good_arms_list = good_arms['Arm_ID'].values.tolist()
# print(good_arms_list)
# good_arms_data = dataset.query('Arm_ID in @good_arms_list')
#
# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', None,
#                        'display.precision', 3):
#     print(good_arms_data)
# print(len(good_arms_data))

# good_arms_data.to_csv('/home/ar1/PycharmProjects/Roni_Master/best_4dof_arms.csv')

'''Group by arm id'''
best_4dof_arms = pd.read_csv('best_4dof_arms.csv')
best_4dof_arms = best_4dof_arms.drop(columns='Unnamed: 0')
print(best_4dof_arms.keys())

grouped_best4dof_data = best_4dof_arms.groupby(['Arm_ID']).agg(
    Success_Rates=pd.NamedAgg(column="Success", aggfunc='mean'),
    Min_Manipulability=pd.NamedAgg(column='Manipulability - mu', aggfunc='min'),
    Max_Manipulability=pd.NamedAgg(column='Manipulability - mu', aggfunc='max'),
    Manipulability_Rates=pd.NamedAgg(column='Manipulability - mu', aggfunc='mean'),
    MaxSum_Mid_joint_proximity=pd.NamedAgg(column='Sum Mid joint proximity- all joints', aggfunc='max')).reset_index()

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3):
    print(grouped_best4dof_data)

''' split arm name to features '''
# new data frame with split value columns
new = grouped_best4dof_data["Arm_ID"].str.split("_", expand=True)
print("Dataframe with splitting Arm's name: ", new)

grouped_best4dof_data["Joint1 type"] = new[1]
grouped_best4dof_data["Joint1 axis"] = new[2]
grouped_best4dof_data["Link1 length"] = new[3] + "." + new[4]

grouped_best4dof_data["Joint2 type"] = new[5]
grouped_best4dof_data["Joint2 axis"] = new[6]
grouped_best4dof_data["Link2 length"] = new[7] + "." + new[8]

grouped_best4dof_data["Joint3 type"] = new[9]
grouped_best4dof_data["Joint3 axis"] = new[10]
grouped_best4dof_data["Link3 length"] = new[11] + "." + new[12]

grouped_best4dof_data["Joint4 type"] = new[13]
grouped_best4dof_data["Joint4 axis"] = new[14]
grouped_best4dof_data["Link4 length"] = new[15] + "." + new[16]

grouped_best4dof_data = grouped_best4dof_data.drop(['Arm_ID'], axis=1)
print("Columns Order", grouped_best4dof_data.columns)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3):
    print(grouped_best4dof_data)

print(grouped_best4dof_data.keys())

best_4dof_joint_conf = grouped_best4dof_data.groupby(['Joint1 type',
                                                      'Joint1 axis', 'Joint2 type', 'Joint2 axis',
                                                      'Joint3 type', 'Joint3 axis', 'Joint4 type',
                                                      'Joint4 axis']).agg(
    Mean_Min_Manipulability=pd.NamedAgg(column='Min_Manipulability', aggfunc='mean'),
    Mean_MaxSum_Mid_joint_proximity=pd.NamedAgg(column='MaxSum_Mid_joint_proximity', aggfunc='mean')).reset_index()

best_4dof_joint_conf['Configuration Index'] = best_4dof_joint_conf.groupby(['Joint1 type',
                                                      'Joint1 axis', 'Joint2 type', 'Joint2 axis',
                                                      'Joint3 type', 'Joint3 axis', 'Joint4 type',
                                                      'Joint4 axis']).ngroup()

best_4dof_joint_conf['1-Mean_Min_Manipulability'] = 1 - best_4dof_joint_conf['Mean_Min_Manipulability']
x = best_4dof_joint_conf['1-Mean_Min_Manipulability']
y = best_4dof_joint_conf['Mean_MaxSum_Mid_joint_proximity']
h = best_4dof_joint_conf['Configuration Index']

sns.scatterplot(x=x, y=y, hue=h, legend='full')
plt.show()

# best_4dof_joint_conf = best_4dof_joint_conf.rename(columns={"index": "Count"})
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3):
    print(best_4dof_joint_conf)



best_4dof_links_conf = grouped_best4dof_data.groupby(
    ['Link1 length', 'Link2 length', 'Link3 length', 'Link4 length']).count().reset_index()
best_4dof_links_conf = best_4dof_links_conf.drop(
    columns=['Success_Rates', 'Min_Manipulability', 'Max_Manipulability', 'Manipulability_Rates',
             'MaxSum_Mid_joint_proximity',
             'Joint1 type',
             'Joint1 axis', 'Joint2 type', 'Joint2 axis',
             'Joint3 type', 'Joint3 axis', 'Joint4 type',
             'Joint4 axis'])
best_4dof_links_conf = best_4dof_links_conf.rename(columns={"index": "Count"})
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3):
    print(best_4dof_links_conf)
