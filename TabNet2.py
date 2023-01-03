import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import optuna
from imblearn.over_sampling import SMOTE
from collections import Counter
from optuna.visualization import plot_contour
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


def split_data(X, y, test_size=0.2, val_size=0.25):
    X = X.to_numpy()
    X = [[float(j) for j in i] for i in X]
    X = np.array(X)
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=1)

    print("X train shape", X_train.shape)
    print("y train shape", y_train.shape)
    print("X val shape", X_val.shape)
    print("y val shape", y_val.shape)
    print("X test shape", X_test.shape)
    print("y test shape", y_test.shape)
    # plot_class_hist(y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_class_hist(y_train, y_val, y_test):
    a = np.unique(np.squeeze(y_test), return_counts=True)
    sns.countplot(x=y_test["class"])
    for v in [0, 1]:
        plt.text(v, a[1][v], str(a[1][v]))
    plt.title('y test')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.show()

    b = np.unique(np.squeeze(y_train), return_counts=True)
    sns.countplot(x=y_train["class"])
    for v in [0, 1]:
        plt.text(v, b[1][v], str(b[1][v]))
    plt.title('y train')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.show()

    c = np.unique(np.squeeze(y_val), return_counts=True)
    sns.countplot(x=y_val["class"])
    for v in [0, 1]:
        plt.text(v, c[1][v], str(c[1][v]))
    plt.title('y valid')
    plt.xlabel('class')
    plt.ylabel('count')
    plt.show()


class TabNet:

    def __init__(self, dataset, imbalanced=False):
        self.estimator = TabNetClassifier()
        self.imbalanced = imbalanced
        if self.imbalanced:
            X = dataset.drop(columns=['Success_Rates', 'class'], axis=1)
            y = dataset[['class']]
            print('Initial dataset shape %s' % y["class"].value_counts())
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            print('Over Resampled dataset shape %s' % y_res["class"].value_counts())
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(X_res, y_res)

        else:
            y = dataset[['class']]
            print('Initial dataset shape %s' % y["class"].value_counts())
            no_sucsses = dataset.loc[dataset['class'] == 0]
            yes_sucsses = dataset.loc[dataset['class'] > 0]
            N_sucsses = no_sucsses.sample(len(yes_sucsses))
            balanced_data = pd.concat([yes_sucsses, N_sucsses]).reset_index(drop=True)
            balanced_data_shuffled = balanced_data.sample(frac=1).reset_index(drop=True)
            X = balanced_data_shuffled.drop(columns=['Success_Rates', 'class'], axis=1)
            y = balanced_data_shuffled[['class']]
            print('Under Resampled dataset shape %s' % y["class"].value_counts())
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(X, y)

    def initial_model(self):
        clf = TabNetClassifier(n_steps=1,
                               optimizer_fn=torch.optim.Adam,
                               optimizer_params=dict(lr=2e-2),
                               scheduler_params={"step_size": 50,  # how to use learning rate scheduler
                                                 "gamma": 0.9},
                               scheduler_fn=torch.optim.lr_scheduler.StepLR,
                               mask_type='entmax',  # "sparsemax",
                               lambda_sparse=0,  # don't penalize for sparser attention

                               )
        max_epochs = 1000
        clf.fit(
            X_train=self.X_train, y_train=np.squeeze(self.y_train),
            eval_set=[(self.X_val, np.squeeze(self.y_val))],
            eval_metric=['auc', 'accuracy'],
            max_epochs=max_epochs,
            patience=50,  # please be patient ^^
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=1,
            drop_last=False,
        )
        plt.plot(clf.history['val_0_auc'][5:])
        plt.title('val_0_auc')
        plt.ylabel('auc')
        plt.xlabel('Epoch')

        # '''Predict Results'''
        preds = clf.predict(self.X_test)
        # print('Classification Report: ')
        # print(classification_report(self.y_test, preds))
        print('Test AUC Score: ')
        print(roc_auc_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.0%')
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()

    def objective(self, trial):
        """Define the objective function"""
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_d = trial.suggest_int("n_d", 44, 64, step=5)
        n_steps = trial.suggest_int("n_steps", 3, 10, step=1)
        gamma = trial.suggest_float("gamma", 1., 2., step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 5)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
        tabnet_params = dict(n_d=n_d, n_a=n_d, n_steps=n_steps, gamma=gamma,
                             lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                             mask_type=mask_type, n_shared=n_shared,
                             scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                             verbose=0, )

        # Fit the model
        optuna_model = TabNetClassifier(**tabnet_params)
        optuna_model.fit(X_train=self.X_train, y_train=np.squeeze(self.y_train),
                         eval_set=[(self.X_val, np.squeeze(self.y_val))])

        # Make predictions
        y_pred = optuna_model.predict(self.X_test)

        # Evaluate predictions
        auc = roc_auc_score(self.y_test, y_pred)
        print("auc score", auc)
        return auc

    def hyper_parameter_tuning(self):
        study = optuna.create_study(direction="maximize", study_name='TabNet optimization')
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
        plt.show()
        optuna.visualization.matplotlib.plot_intermediate_values(study)
        plt.show()
        # optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        # plt.show()
        # optuna.visualization.matplotlib.plot_contour(study)
        # plt.show()
        # optuna.visualization.matplotlib.plot_slice(study)
        # plt.show()
        # optuna.visualization.matplotlib.plot_param_importances(study)
        # plt.show()

        tab_params = trial.params

        return tab_params

    def plot_performance(self, best_params):
        estimator = TabNetClassifier(**best_params)
        '''Fit Results'''
        estimator.fit(X_train=self.X_train, y_train=np.squeeze(self.y_train),
                      eval_set=[(self.X_val, np.squeeze(self.y_val))],
                      eval_metric=['auc', 'accuracy'])

        # plot auc value
        # results = estimator.evals_result()
        # epochs = len(results['validation_0']['auc'])
        # x_axis = range(0, epochs)
        # fig, ax = plt.subplots()
        # ax.plot(x_axis, results['validation_0']['auc'], label='Train')
        # ax.plot(x_axis, results['validation_1']['auc'], label='Validation')
        # ax.legend()
        # plt.ylabel('AUC')
        # plt.title('TabNet AUC')
        # plt.show()
        #
        # # plot error #(wrong cases)/#(all cases)
        # epochs = len(results['validation_0']['error'])
        # x_axis = range(0, epochs)
        # fig, ax = plt.subplots()
        # ax.plot(x_axis, results['validation_0']['error'], label='Train')
        # ax.plot(x_axis, results['validation_1']['error'], label='Validation')
        # ax.legend()
        # plt.ylabel('Error #(wrong cases)/#(all cases)')
        # plt.title('TabNet Error')
        # plt.show()

        '''Predict Results'''
        preds = estimator.predict(self.X_test)
        # print('Classification Report: ')
        # print(classification_report(self.y_test, preds))
        print('Test AUC Score: ')
        print(roc_auc_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.0%')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()

        # errors = np.squeeze(self.y_test) - preds
        # error_df = pd.DataFrame(errors, columns=["error"])
        # amount = [error_df['error'].loc[error_df['error'] == -1].count(),
        #           error_df['error'].loc[error_df['error'] == 0].count(),
        #           error_df['error'].loc[error_df['error'] == 1].count()]
        # data = {'clusters error': [*range(-1, 2, 1)], 'amount': amount}
        # df = pd.DataFrame(data)
        #
        # plt.bar(x=df['clusters error'], height=df['amount'])
        # plt.title('errors - True-Predictions')
        # plt.xlabel('errors [Clusters]')
        # plt.ylabel('count')

        # xgb.plot_tree(estimator)
        # plt.show()
        # xgb.plot_tree(estimator, num_trees=4)
        # plt.show()


if __name__ == '__main__':
    data = pd.read_csv('6dof_disc_and_classification.csv')
    data.drop('Unnamed: 0', inplace=True, axis=1)
    data['class'] = np.where(data['Success_Rates'] > 0, 1, 0)

    tab = TabNet(dataset=data, imbalanced=True)
    best_params = tab.hyper_parameter_tuning()
    tab.plot_performance(best_params)
    # tab.initial_model()


#roni