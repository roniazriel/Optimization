from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
# import torch
import xgboost as xgb
from numpy import mean
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, roc_curve, \
    auc, f1_score, fbeta_score, make_scorer, log_loss, recall_score, matthews_corrcoef, accuracy_score
import optuna
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks

'''Ignore Warnings'''
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def split_data(df, to_balance, over_sampling=True, method=1,
               plot_path='/home/ar1/Desktop/plots/'):
    """ Split to datasets - A certain arm will appear only in one set, the prediction will be for all ten points """
    df["class"] = np.where(df['Success'] > 0, 1, 0)
    df.drop(columns=['Unnamed: 0', 'Move duration', 'Success', 'Manipulability - mu',
                     'Mid joint proximity', 'Max Mid joint proximity', 'Sum Mid joint proximity- all joints'],
            inplace=True)

    grapes_cords = [[1.2, 0, 1.2],
                    [1.05, 0, 1.28],
                    [1.1, 0, 1.35],
                    [0.9, 0, 1.47],
                    [1.2, 0, 1.5],
                    [1.05, 0, 1.6],
                    [0.85, 0, 1.7],
                    [1.1, 0, 1.75],
                    [1.2, 0, 1.8],
                    [0.7, 0, 1.8]]

    conditions = [
        (df['Point number'] == 1),
        (df['Point number'] == 2),
        (df['Point number'] == 3),
        (df['Point number'] == 4),
        (df['Point number'] == 5),
        (df['Point number'] == 6),
        (df['Point number'] == 7),
        (df['Point number'] == 8),
        (df['Point number'] == 9),
        (df['Point number'] == 10),
    ]

    # create a list of the values we want to assign for each condition
    values = [i[0] for i in grapes_cords]
    # [i[0] for i in grapes_cords]
    values1 = [i[2] for i in grapes_cords]
    # create a new column and use np.select to assign values to it using our lists as arguments
    df['PointX'] = np.select(conditions, values)
    df['PointY'] = 0
    df['PointZ'] = np.select(conditions, values1)

    ''' Drop unnecessary features: '''
    df.drop(columns=['PointY', 'Point number', 'Joint1 axis', 'Link1 length', 'Joint1 type'], inplace=True)

    '''One Hot encoder to categorical features'''

    df = pd.get_dummies(df
                        , columns=['Joint2 type', 'Joint2 axis', 'Joint3 type', 'Joint3 axis', 'Joint4 type',
                                   'Joint4 axis',
                                   'Joint5 type', 'Joint5 axis', 'Joint6 type', 'Joint6 axis']
                        , drop_first=True
                        )

    dup = df['Arm_ID'].drop_duplicates().reset_index()
    dup.drop(columns='index', inplace=True)
    to_test = dup.sample(frac=0.4)
    to_valid = to_test.sample(frac=0.5)
    to_test = to_test[~to_test.Arm_ID.isin(to_valid.Arm_ID)]

    """ Check the process """
    check = pd.concat([to_valid, to_test]).reset_index(drop=True)
    print(check)
    print("Is there an overlap between test and validation?: ", True in check.duplicated())

    test_set = df[df.Arm_ID.isin(to_test.Arm_ID)]
    val_set = df[df.Arm_ID.isin(to_valid.Arm_ID)]
    train_set = df[~df.Arm_ID.isin(to_valid.Arm_ID) & ~df.Arm_ID.isin(to_test.Arm_ID)]

    test_set.drop(columns='Arm_ID', inplace=True)
    val_set.drop(columns='Arm_ID', inplace=True)
    train_set.drop(columns='Arm_ID', inplace=True)

    # train = pd.concat([train_set,val_set]).reset_index()
    y_train = train_set[['class']]
    X_train = train_set.drop(columns=['class'])
    y_val = val_set[['class']]
    X_val = val_set.drop(columns=['class'])
    y_test = test_set[['class']]
    X_test = test_set.drop(columns=['class'])

    if to_balance:
        if over_sampling:
            if method == 0:
                rmo = RandomOverSampler(random_state=42)
                X_res, y_res = rmo.fit_resample(X_train, y_train)
            elif method == 1:
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_train, y_train)
            elif method == 2:
                bsm = BorderlineSMOTE(random_state=42)
                X_res, y_res = bsm.fit_resample(X_train, y_train)
            elif method == 3:
                ad = ADASYN(random_state=42)
                X_res, y_res = ad.fit_resample(X_train, y_train)
            else:
                print("Wrong method number!")
        else:
            if method == 1:
                Rm = RandomUnderSampler(random_state=42)
                X_res, y_res = Rm.fit_resample(X_train, y_train)
            elif method == 2:
                Nm = NearMiss(version=3)
                X_res, y_res = Nm.fit_resample(X_train, y_train)
            elif method == 3:
                tml = TomekLinks()
                X_res, y_res = tml.fit_resample(X_train, y_train)
            else:
                print("Wrong method number!")

        print('Resampled train dataset shape %s' % y_res["class"].value_counts())
        temp = X_res.assign(target=y_res["class"].values)
        temp = temp.sample(frac=1, axis=0).reset_index(drop=True)
        temp.rename(columns={"target": "class"}, inplace=True)
        X_res = temp.drop(columns=['class'], axis=1)
        y_res = temp[['class']]
        plot_class_hist(y_train, y_res, y_val, y_test, plot_path=plot_path, over_sampling=over_sampling,
                        method=method)
        X_train = X_res
        y_train = y_res
    else:
        plot_class_hist(y_train=y_train, y_new_train=None, y_val=y_val, y_test=y_test, plot_path=plot_path,
                        over_sampling=over_sampling,
                        method=method)
        print('Train dataset class Distribution %s' % y_train["class"].value_counts())

    print('Test dataset class Distribution %s' % y_test["class"].value_counts())
    print("X train shape", X_train.shape)
    print("y train shape", y_train.shape)
    print("X val shape", X_val.shape)
    print("y val shape", y_val.shape)
    print("X test shape", X_test.shape)
    print("y test shape", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_class_hist(y_train, y_new_train, y_val, y_test, over_sampling, method,
                    plot_path='/home/ar1/Desktop/plots/'):
    if y_new_train is None:
        temp_train = y_train[['class']]
        temp_test = y_test[['class']]
        temp_val = y_val[['class']]

        temp_train["Dataset"] = "Train"
        temp_test["Dataset"] = "Test"
        temp_val["Dataset"] = "Validation"
        visualization1 = pd.concat([temp_train, temp_test, temp_val], ignore_index=True)
        fig, ax = plt.subplots()
        sns.countplot(data=visualization1, x='Dataset', hue='class', palette='mako', ax=ax).set(
            title='Datasets after '
                  'Balancing')

    else:
        temp_train = y_new_train[['class']]
        temp_old_train = y_train[['class']]
        temp_train["Dataset"] = "Train"
        temp_old_train["Dataset"] = "Imbalanced Train"
        visualization2 = pd.concat([temp_train, temp_old_train], ignore_index=True)

        temp_test = y_test[['class']]
        temp_val = y_val[['class']]
        temp_train["Dataset"] = "Train"
        temp_test["Dataset"] = "Test"
        temp_val["Dataset"] = "Validation"

        visualization1 = pd.concat([temp_train, temp_test, temp_val], ignore_index=True)

        fig, ax = plt.subplots(1, 2)
        sns.countplot(data=visualization1, x='Dataset', hue='class', palette='mako', ax=ax[1]).set(
            title='Datasets after '
                  'Balancing')
        sns.countplot(data=visualization2, x='Dataset', hue='class', palette='mako', ax=ax[0]).set(
            title='Train dataset - '
                  'Balancing')

    fig.suptitle("Class Distribution")
    fig.tight_layout()
    plt.savefig(plot_path + 'Class Distribution' + str(over_sampling) + str(method) + ".png")


class XGB:

    def __init__(self, dataset, to_balance=False, over_sampling=False, method=1,
                 plot_path='/home/ar1/Desktop/plots/'):
        self.estimator = xgb.XGBClassifier(random_state=13)
        self.to_balance = to_balance
        self.over_sampling = over_sampling
        self.method = method
        self.plot_path = plot_path
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(df=dataset,
                                                                                                  to_balance=self.to_balance,
                                                                                                  over_sampling=self.over_sampling,
                                                                                                  method=self.method,
                                                                                                  plot_path=plot_path)

    def baseline(self, k=5):
        model = xgb.XGBClassifier(random_state=13)
        self.model_evaluation(best_params={'random_state': 13})

    def objective(self, trial):
        """Define the objective function"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'logloss'
        }

        if not self.to_balance:
            params.update({'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 300, step=50)})

            # Fit the model
        optuna_model = xgb.XGBClassifier(**params)
        optuna_model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = optuna_model.predict(self.X_val)
        y_prob = optuna_model.predict_proba(self.X_val)

        # Evaluate predictions
        f2 = fbeta_score(self.y_val, y_pred, average='weighted', beta=2)
        lg = log_loss(self.y_val, y_prob)

        print("Accuracy : %.4g" % accuracy_score(self.y_val, y_pred))
        print("Log Loss : %.4g" % lg)
        print("AUC Score: %f" % roc_auc_score(self.y_val, y_pred))
        print("Recall Score: %f" % recall_score(self.y_val, y_pred))
        print("matthews_corrcoef Score: %f" % matthews_corrcoef(self.y_val, y_pred))
        print("f2 score", f2)

        return matthews_corrcoef(self.y_val, y_pred)

    def hyper_parameter_tuning(self):
        study = optuna.create_study(direction="maximize", study_name='XGB optimization')
        study.optimize(self.objective, timeout=6 * 60)  # 6 hours

        print('Number of finished trials: {}'.format(len(study.trials)))
        print('Best trial:')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))
        print('  Params: ')

        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        # plot best trial
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(
            self.plot_path + 'optimization_history' + str(self.over_sampling) + str(
                self.method) + ".png")
        # optuna.visualization.matplotlib.plot_intermediate_values(study) plt.savefig(
        # '/home/ar1/Desktop/plots/intermediate_values' + str(self.over_sampling) + str(self.method) + ".png")
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.savefig(self.plot_path + 'parallel_coordinate' + str(self.over_sampling) + str(self.method) +
                    ".png")
        optuna.visualization.matplotlib.plot_contour(study)
        plt.savefig(self.plot_path + 'contour' + str(self.over_sampling) + str(self.method) +
                    ".png")
        optuna.visualization.matplotlib.plot_slice(study)
        plt.savefig(self.plot_path + 'slice' + str(self.over_sampling) + str(self.method) +
                    ".png")
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(self.plot_path + 'param_importance' + str(self.over_sampling) + str(self.method) +
                    ".png")

        xgb_params = trial.params
        return xgb_params

    def plot_performance(self, fitted_estimator):
        '''Fit Results'''
        preds = fitted_estimator.predict(self.X_test)
        preds_prob = fitted_estimator.predict_proba(self.X_test)  # [:,1];
        plt.clf()

        # plot roc curve
        fpr, tpr, _ = roc_curve(self.y_test, preds_prob[:, 1])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="ROC curve (area = " + str(round(auc_score, 3)) + " )")
        # create ROC curve
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) curve - " + "Test")
        plt.legend()
        plt.savefig(
            self.plot_path + 'fit (ROC) curve' + str(self.over_sampling) + str(self.method) +
             ".png")

        plt.clf()

        # plot error #(wrong cases)/#(all cases)
        results = fitted_estimator.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        # fig, ax = plt.subplots()
        plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
        plt.legend()
        plt.ylabel('logloss')
        plt.title('XGBoost logloss')
        plt.savefig(self.plot_path + 'fit logloss' + str(self.over_sampling) + str(self.method) +
                    ".png")
        plt.clf()

        '''Predict Results'''
        print('Classification Report: ')
        print(classification_report(self.y_test, preds))
        report_dict = classification_report(self.y_test, preds, output_dict=True)
        pd.DataFrame(report_dict).to_csv(self.plot_path + 'classification_report' + str(self.over_sampling) +
                                         str(self.method) + ".csv")

        cm = confusion_matrix(self.y_test, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.0%')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.savefig(
            self.plot_path + 'fit Confusion Matrix' + str(self.over_sampling) + str(self.method) +
            ".png")
        plt.clf()

        # xgb.plot_tree(estimator)
        # plt.show()
        # xgb.plot_tree(estimator, num_trees=4)
        # plt.show()

    def model_evaluation(self, best_params):
        self.estimator = xgb.XGBClassifier(**best_params)

        self.estimator.fit(self.X_train, self.y_train,
                           eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)])

        preds = self.estimator.predict(self.X_test)
        preds_prob = self.estimator.predict_proba(self.X_test)
        print('Test Results:')
        print("Test Accuracy : %.4g" % accuracy_score(self.y_test, preds))
        print("Log Loss : %.4g" % log_loss(self.y_test, preds_prob))
        print("AUC Score (Test): %f" % roc_auc_score(self.y_test, preds))
        print("Recall Score (Test): %f" % recall_score(self.y_test, preds))
        print("f2 Score (Test): %f" % fbeta_score(self.y_test, preds, beta=2, average='weighted'))
        print("MCC Score CV: %f" % matthews_corrcoef(self.y_test, preds))

        self.plot_performance(self.estimator)


if __name__ == '__main__':
    df = pd.read_csv('6dof_dataset.csv')

    # results = pd.DataFrame(columns=['Learning Task', 'lb', 'ub', 'over sampling?', 'method'])
    # methods = [1, 2, 3]
    # over_sampling = [True, False]

    # lb = 3
    # ub = 9
    #     n_class = data['class'].nunique()
    #     if t == 'Binary':
    #         multi = False
    #     else:
    #         multi = True
    #     for v in over_sampling:
    #         for i in methods:
    #             print('  Oversampling?: {}'.format(v))
    #             print('  method: {}'.format(i))
    #             print('  Learning task: {}'.format(t))
    #             print("1=SMOTE, 2=Borderline SMOTE, 3=ADASYN / 1=Random, 2=NearMiss, 3=Tomek's links")
    #             xg = XGB(dataset=data, over_sampling=v, method=i, multiclass=multi,
    #                      plot_path="C:\\Users\\azrie\\Desktop\\plots\\")
    #             best_params = xg.hyper_parameter_tuning()
    #             xg.model_evaluation(best_params)
    #             results.append({'Learning Task': t, 'lb': lb, 'ub': ub, 'over sampling?': v, 'method': i},
    #                            ignore_index=True)
    #
    # results.to_csv("Results_comparison.csv")
    # xg = XGB(dataset=data, over_sampling=True, method=1, multiclass=False,
    #          plot_path="C:\\Users\\azrie\\Desktop\\plots\\")
    # best_params = xg.hyper_parameter_tuning()
    # xg.plot_performance(best_params)
    # xg.model_evaluation(best_params)

    xg = XGB(dataset=df, to_balance=False, over_sampling=False, method=1)
    # xg.baseline()
    best_params = xg.hyper_parameter_tuning()
    xg.model_evaluation(best_params)
