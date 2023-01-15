from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
import pandas as pd
import numpy as np


def resample_data(X_train, y_train, over_sampling=True, method=1,
               plot_path='/home/ar1/Desktop/plots/'):
    print('Initial dataset shape %s' % y_train["class"].value_counts())
    print("X train: " ,X_train)
    print("y train: " ,y_train)

    if over_sampling:
        if method == 1:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 2:
            bsm = BorderlineSMOTE(random_state=42)
            X_res, y_res = bsm.fit_resample(X_train, y_train)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 3:
            ad = ADASYN(random_state=42)
            X_res, y_res = ad.fit_resample(X_train, y_train)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        else:
            print("Wrong method number!")

    else:
        if method == 1:
            Rm = RandomUnderSampler(random_state=42)
            X_res, y_res = Rm.fit_resample(X_train, y_train)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 2:
            Nm = NearMiss(version=3)
            X_res, y_res = Nm.fit_resample(X_train, y_train)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 3:
            tml = TomekLinks()
            X_res, y_res = tml.fit_resample(X_train, y_train)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        else:
            print("Wrong method number!")

    print("New X train ", X_res)
    print("New y train shape", y_res)

    return X_res, y_res


if __name__ == '__main__':
    df = pd.read_csv('6dof_disc_and_classification.csv')
    df.drop('Unnamed: 0', inplace=True, axis=1)
    ten = df.loc[df['Success_Rates'] == 10]
    zero = df.loc[df['Success_Rates'] == 0]
    new_df = pd.concat([ten.sample(10), zero.sample(20)]).reset_index()
    new_df["class"] = np.where(new_df['Success_Rates'] > 0, 1, 0)
    # new_df.to_csv("must.csv")
    X = new_df.drop(columns=['Success_Rates', 'class'], axis=1)
    y = new_df[['class']]
    x_res,y_res = resample_data(X_train=X,y_train=y,over_sampling=True,method=1)

    data_res = pd.concat([x_res,y_res],axis=1).reset_index(drop=True)
    data_res.drop(columns=['index'], inplace=True)
    # data_res.to_csv("testtttt.csv")
    print("datares ,",data_res)
    new_df.drop(columns=['Success_Rates','index'], inplace=True)
    print("new_df ,", new_df)
    new_x = pd.concat([new_df,data_res]).drop_duplicates(keep=False).reset_index()
    # new_x.drop(columns=['level_0', 'index'], inplace=True)
    # new_x = pd.concat([x_res, X]).duplicated(keep=False)
    print(new_x)
