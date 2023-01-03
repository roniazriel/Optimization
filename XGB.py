import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import optuna
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from sklearn.decomposition import PCA
from collections import Counter
from optuna.visualization import plot_contour
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


def split_data(X, y, test_size=0.2, val_size=0.25, over_sampling=True, method=1):
    print('Initial dataset shape %s' % y["class"].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_train)
    label = y_train[['class']]
    label['class_name'] = np.where(y_train['class'] == 0, "Not Reached", "Reached")
    fig = px.scatter(components, x=0, y=1, color=label['class'],color_continuous_scale=px.colors.qualitative.Dark24)
    fig.update_layout(title='PCA -Train Data before resample')
    fig.write_image("/home/ar1/Desktop/plots/PCA before resample"+str(over_sampling)+str(method)+".png")

    if over_sampling:
        if method == 1:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 2:
            bsm = BorderlineSMOTE(random_state=42)
            X_res, y_res = bsm.fit_resample(X, y)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 3:
            ad = ADASYN(random_state=42)
            X_res, y_res = ad.fit_resample(X, y)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        else:
            print("Wrong method number!")

    else:
        if method == 1:
            Rm = RandomUnderSampler(random_state=42)
            X_res, y_res = Rm.fit_resample(X, y)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 2:
            Nm = NearMiss(version=3)
            X_res, y_res = Nm.fit_resample(X, y)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        elif method == 3:
            tml = TomekLinks()
            X_res, y_res = tml.fit_resample(X, y)
            print('Over Resampled train dataset shape %s' % y_res["class"].value_counts())
        else:
            print("Wrong method number!")

    X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=val_size, random_state=1)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_res)
    label = y_res[['class']]
    label['class_name'] = np.where(y_res['class'] == 0, "Not Reached", "Reached")
    fig = px.scatter(components, x=0, y=1, color=label['class'],color_continuous_scale=px.colors.qualitative.Dark24)
    fig.update_layout(title='PCA -Dataset after resample')
    fig.write_image("/home/ar1/Desktop/plots/PCA after resample"+str(over_sampling)+str(method)+".png")

    print("X train shape", X_train.shape)
    print("y train shape", y_train.shape)
    print("X val shape", X_val.shape)
    print("y val shape", y_val.shape)
    print("X test shape", X_test.shape)
    print("y test shape", y_test.shape)
    plot_class_hist(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_class_hist(y_train, y_val, y_test,over_sampling=True, method=1):
    a = np.unique(np.squeeze(y_test), return_counts=True)
    sns.countplot(x=y_test["class"])
    for v in [0, 1]:
        plt.text(v, a[1][v], str(a[1][v]))
    plt.title('y test')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.savefig('/home/ar1/Desktop/plots/y test' + str(over_sampling) + str(method) + ".png")

    b = np.unique(np.squeeze(y_train), return_counts=True)
    sns.countplot(x=y_train["class"])
    for v in [0, 1]:
        plt.text(v, b[1][v], str(b[1][v]))
    plt.title('y train')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.savefig('/home/ar1/Desktop/plots/y train' + str(over_sampling) + str(method) + ".png")

    c = np.unique(np.squeeze(y_val), return_counts=True)
    sns.countplot(x=y_val["class"])
    for v in [0, 1]:
        plt.text(v, c[1][v], str(c[1][v]))
    plt.title('y valid')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.savefig('/home/ar1/Desktop/plots/y valid' + str(over_sampling) + str(method) + ".png")


class XGB:

    def __init__(self, dataset, over_sampling=False, method=1):
        self.estimator = xgb.XGBClassifier()
        self.over_sampling = over_sampling
        self.method = method
        X = dataset.drop(columns=['Success_Rates', 'class'], axis=1)
        y = dataset[['class']]
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(X, y,
                                                                                                  over_sampling=self.over_sampling,
                                                                                                  method=self.method)

    def objective(self, trial):
        """Define the objective function"""

        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'auc',
            'use_label_encoder': False
        }

        # Fit the model
        optuna_model = xgb.XGBClassifier(**params)
        optuna_model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = optuna_model.predict(self.X_test)

        # Evaluate predictions
        auc = roc_auc_score(self.y_test, y_pred)
        print("auc score", auc)
        return auc

    def hyper_parameter_tuning(self):
        params = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'booster': ['gbtree', 'gblinear'],
            'gamma': [0, 0.5, 1],
            'reg_alpha': [0, 0.5, 1],
            'reg_lambda': [0.5, 1, 5],
            'base_score': [0.2, 0.5, 1],
            'scale_pos_weight': []
            # A typical value to consider: sum(negative instances) / sum(positive instances).
        }

        study = optuna.create_study(direction="maximize", study_name='XGB optimization')
        study.optimize(self.objective, timeout=6 * 60)  # 5 hours

        print('Number of finished trials: {}'.format(len(study.trials)))
        print('Best trial:')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))
        print('  Params: ')

        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        # plot best trial
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig('/home/ar1/Desktop/plots/optimization_history' + str(self.over_sampling) + str(self.method) + ".png")
        # optuna.visualization.matplotlib.plot_intermediate_values(study)
        # plt.savefig('/home/ar1/Desktop/plots/intermediate_values' + str(self.over_sampling) + str(self.method) + ".png")
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.savefig('/home/ar1/Desktop/plots/parallel_coordinate' + str(self.over_sampling) + str(self.method) + ".png")
        optuna.visualization.matplotlib.plot_contour(study)
        plt.savefig('/home/ar1/Desktop/plots/contour' + str(self.over_sampling) + str(self.method) + ".png")
        optuna.visualization.matplotlib.plot_slice(study)
        plt.savefig('/home/ar1/Desktop/plots/slice' + str(self.over_sampling) + str(self.method) + ".png")
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig('/home/ar1/Desktop/plots/param_importance' + str(self.over_sampling) + str(self.method) + ".png")

        xgb_params = trial.params
        return xgb_params

    def plot_performance(self, best_params):
        estimator = xgb.XGBClassifier(**best_params)
        '''Fit Results'''
        estimator.fit(self.X_train, self.y_train,
                      eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                      eval_metric=['auc', 'error'])

        # plot auc value
        results = estimator.evals_result()
        epochs = len(results['validation_0']['auc'])
        x_axis = range(0, epochs)
        # fig, ax = plt.subplots()
        plt.plot(x_axis, results['validation_0']['auc'], label='Train')
        plt.plot(x_axis, results['validation_1']['auc'], label='Validation')
        plt.legend()
        plt.ylabel('AUC')
        plt.title('XGBoost AUC')
        plt.savefig('/home/ar1/Desktop/plots/fit AUC' + str(self.over_sampling) + str(self.method) + ".png")

        # plot error #(wrong cases)/#(all cases)
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        # fig, ax = plt.subplots()
        plt.plot(x_axis, results['validation_0']['error'], label='Train')
        plt.plot(x_axis, results['validation_1']['error'], label='Validation')
        plt.legend()
        plt.ylabel('Error #(wrong cases)/#(all cases)')
        plt.title('XGBoost Error')
        plt.savefig('/home/ar1/Desktop/plots/fit error' + str(self.over_sampling) + str(self.method) + ".png")

        '''Predict Results'''
        preds = estimator.predict(self.X_test)
        print('Classification Report: ')
        print(classification_report(self.y_test, preds))
        report_dict = classification_report(self.y_test, preds, output_dict=True)
        pd.DataFrame(report_dict).to_csv('/home/ar1/Desktop/plots/classification_report.csv')
        print('Test AUC Score: ')
        print(roc_auc_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.0%')
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.savefig('/home/ar1/Desktop/plots/fit Confusion Matrix' + str(self.over_sampling) + str(self.method) + ".png")

        errors = np.squeeze(self.y_test) - preds
        error_df = pd.DataFrame(errors, columns=["error"])
        amount = [error_df['error'].loc[error_df['error'] == -1].count(),
                  error_df['error'].loc[error_df['error'] == 0].count(),
                  error_df['error'].loc[error_df['error'] == 1].count()]
        data = {'clusters error': [*range(-1, 2, 1)], 'amount': amount}
        df = pd.DataFrame(data)

        plt.bar(x=df['clusters error'], height=df['amount'])
        plt.title('errors - True-Predictions')
        plt.xlabel('errors [Clusters]')
        plt.ylabel('count')
        plt.savefig('/home/ar1/Desktop/plots/errors bar' + str(self.over_sampling) + str(self.method) + ".png")

        # xgb.plot_tree(estimator)
        # plt.show()
        # xgb.plot_tree(estimator, num_trees=4)
        # plt.show()


if __name__ == '__main__':
    data = pd.read_csv('6dof_disc_and_classification.csv')
    data.drop('Unnamed: 0', inplace=True, axis=1)
    data['class'] = np.where(data['Success_Rates'] > 0, 1, 0)
    methods = [1, 2, 3]
    over_sampling = [True, False]

    # for v in over_sampling:
    #     for i in methods:
    #         print('  Oversampling?: {}'.format(v))
    #         print('  method: {}'.format(i))
    #         print("1=SMOTE, 2=Borderline SMOTE, 3=ADASYN / 1=Random, 2=NearMiss, 3=Tomek's links")
    #         xg = XGB(dataset=data, over_sampling=v, method=i)
    #         best_params = xg.hyper_parameter_tuning()
    #         xg.plot_performance(best_params)

    xg = XGB(dataset=data, over_sampling=False, method=1)
    best_params = xg.hyper_parameter_tuning()
    xg.plot_performance(best_params)
