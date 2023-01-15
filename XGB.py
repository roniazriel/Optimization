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
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold


def learning_task(data, task, lb=None, ub=None):
    if task == 'Binary':
        data['class'] = np.where(data['Success_Rates'] > 0, 1, 0)
        return data, 'Binary'
    elif task == 'Multi':
        # data.rename(columns={"Success_Rates": "class"}, inplace=True)
        data["class"] = data["Success_Rates"]
        return data, 'Multi'
    elif task == '3 class':
        col = 'Success_Rates'
        conditions = [data[col] >= ub, (data[col] < ub) & (data[col] > lb), data[col] <= lb]
        choices = [2, 1, 0]
        data["class"] = np.select(conditions, choices, default=np.nan)
        return data, '3 class: ' + str(lb) + "," + str(ub)
    else:
        print("Invalid task!")


def split_data(X, y, imbalanced, learn_name, test_size=0.2, val_size=0.25, over_sampling=True, method=1,
               plot_path='/home/ar1/Desktop/plots/'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=5)
    print('Initial train dataset shape %s' % y_train["class"].value_counts())
    if not imbalanced:
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
        plot_class_hist(y_train, y_res, y_val, y_test, plot_path=plot_path, learn_name=learn_name,
                        over_sampling=over_sampling,
                        method=method)
        X_train = X_res
        y_train = y_res

    print("X train shape", X_train.shape)
    print("y train shape", y_train.shape)
    print("X val shape", X_val.shape)
    print("y val shape", y_val.shape)
    print("X test shape", X_test.shape)
    print("y test shape", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_class_hist(y_train, y_new_train, y_val, y_test, over_sampling, method, learn_name,
                    plot_path='/home/ar1/Desktop/plots/'):
    temp_train = y_new_train[['class']]
    temp_old_train = y_train[['class']]
    temp_test = y_test[['class']]
    temp_val = y_val[['class']]

    temp_train["Dataset"] = "Train"
    temp_old_train["Dataset"] = "Imbalanced Train"
    temp_test["Dataset"] = "Test"
    temp_val["Dataset"] = "Validation"
    visualization1 = pd.concat([temp_train, temp_test, temp_val], ignore_index=True)
    visualization2 = pd.concat([temp_train, temp_old_train], ignore_index=True)

    fig, ax = plt.subplots(1, 2)
    sns.countplot(data=visualization1, x='Dataset', hue='class', palette='mako', ax=ax[1]).set(title='Datasets after '
                                                                                                     'Balancing')
    sns.countplot(data=visualization2, x='Dataset', hue='class', palette='mako', ax=ax[0]).set(title='Train dataset - '
                                                                                                     'Balancing')
    fig.suptitle("Class Distribution")
    fig.tight_layout()
    plt.savefig(plot_path + 'Class Distribution' + str(over_sampling) + str(method) +
                " " + str(learn_name) + ".png")


def plot_roc_curve(y_test, y_pred, over_sampling, method, learn_name, save=False, dataset='Validation',
                   plot_path='/home/ar1/Desktop/plots/'):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))
    # y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    #    plt.figure(dpi=300)
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
             color="pink", linestyle="-.", linewidth=4, )

    plt.plot(fpr["macro"], tpr["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
             color="purple", linestyle="-.", linewidth=4, )

    colors = cycle(["gray", "green", "blue", "yellow", "red", 'black', 'brown', 'goldenrod', 'gold',
                    'aqua', 'violet', 'darkslategray', 'mistyrose', 'darkorange', 'tan'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle="--",
                 label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]), )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curve - " + dataset)
    plt.legend()
    if save:
        plt.savefig(
            plot_path + 'ROC Curve' + str(over_sampling) + str(
                method) + " " + str(learn_name) + ".png")
    else:
        plt.show()


class XGB:

    def __init__(self, dataset, learning_task, imbalanced=False, over_sampling=False, method=1, multiclass=False,
                 num_class=2,
                 plot_path='/home/ar1/Desktop/plots/'):
        self.estimator = xgb.XGBClassifier(random_state=13)
        self.imbalanced = imbalanced
        self.over_sampling = over_sampling
        self.learning_task = learning_task
        self.method = method
        self.plot_path = plot_path
        self.multiclass = multiclass
        self.num_class = num_class
        self.X = dataset.drop(columns=['Success_Rates', 'class'], axis=1)
        self.y = dataset[['class']]
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(self.X, self.y,
                                                                                                  imbalanced=self.imbalanced,
                                                                                                  over_sampling=self.over_sampling,
                                                                                                  method=self.method,
                                                                                                  plot_path=plot_path,
                                                                                                  learn_name=self.learning_task)

    def baseline(self, k=5):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=1)
        # model = xgb.XGBClassifier(random_state=13)
        ftwo_scorer = make_scorer(fbeta_score, beta=2)
        matthews = make_scorer(matthews_corrcoef)
        scoring = {'f2_score': ftwo_scorer, 'matthews': matthews, 'accuracy': 'accuracy', 'roc_auc': 'roc_auc',
                   'recall': 'recall'}

        scores = cross_validate(self.estimator, X_train, y_train, cv=k,
                                scoring=scoring)

        print('Validation Results:')
        print("Accuracy CV: %.4g" % np.mean(scores['test_accuracy']))
        print("AUC Score CV: %f" % np.mean(scores['test_roc_auc']))
        print("Recall Score CV: %f" % np.mean(scores['test_recall']))
        print("f2 Score CV: %f" % np.mean(scores['test_f2_score']))
        print("MCC Score CV: %f" % np.mean(scores['test_matthews']))

        best_params = self.hyper_parameter_tuning()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=1)

        self.model_evaluation(best_params=best_params, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    def objective(self, trial):
        """Define the objective function"""
        if self.multiclass:
            params = {
                'max_depth': trial.suggest_int('max_depth', 1, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
                'eval_metric': 'mlogloss',
                'objective': 'multi:softprob',  # error evaluation for multiclass training
                'num_class': self.num_class
            }
        else:
            params = {
                'max_depth': trial.suggest_int('max_depth', 1, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
                'eval_metric': 'logloss'
            }
        if self.imbalanced:
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
                self.method) + " " + str(self.learning_task) + ".png")
        # optuna.visualization.matplotlib.plot_intermediate_values(study) plt.savefig(
        # '/home/ar1/Desktop/plots/intermediate_values' + str(self.over_sampling) + str(self.method) + ".png")
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.savefig(self.plot_path + 'parallel_coordinate' + str(self.over_sampling) + str(self.method) +
                    " " + str(self.learning_task) + ".png")
        optuna.visualization.matplotlib.plot_contour(study)
        plt.savefig(self.plot_path + 'contour' + str(self.over_sampling) + str(self.method) +
                    " " + str(self.learning_task) + ".png")
        optuna.visualization.matplotlib.plot_slice(study)
        plt.savefig(self.plot_path + 'slice' + str(self.over_sampling) + str(self.method) +
                    " " + str(self.learning_task) + ".png")
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(self.plot_path + 'param_importance' + str(self.over_sampling) + str(self.method) +
                    " " + str(self.learning_task) + ".png")

        xgb_params = trial.params
        return xgb_params

    def plot_performance(self, fitted_estimator):
        '''Fit Results'''
        preds = fitted_estimator.predict(self.X_test)
        preds_prob = fitted_estimator.predict_proba(self.X_test)  # [:,1];

        if self.multiclass:
            plot_roc_curve(self.y_test, preds_prob, dataset='Test set', over_sampling=self.over_sampling,
                           method=self.method,
                           learn_name=self.learning_task, save=True, plot_path=self.plot_path)

            plt.clf()
            # plot error #(wrong cases)/#(all cases)
            results = fitted_estimator.evals_result()
            epochs = len(results['validation_0']['mlogloss'])
            x_axis = range(0, epochs)
            # fig, ax = plt.subplots()
            plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
            plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
            plt.legend()
            plt.ylabel('logloss')
            plt.title('XGBoost logloss')
            plt.savefig(self.plot_path + 'fit logloss' + str(self.over_sampling) + str(self.method) +
                        " " + str(self.learning_task) + ".png")
            plt.clf()

        else:
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
                " " + str(self.learning_task) + ".png")

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
                        " " + str(self.learning_task) + ".png")
            plt.clf()

        '''Predict Results'''
        print('Classification Report: ')
        print(classification_report(self.y_test, preds))
        report_dict = classification_report(self.y_test, preds, output_dict=True)
        pd.DataFrame(report_dict).to_csv(self.plot_path + 'classification_report' + str(self.over_sampling) +
                                         str(self.method) +
                                         " " + str(self.num_class) + " " + "classes" + ".csv")

        cm = confusion_matrix(self.y_test, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.0%')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.savefig(
            self.plot_path + 'fit Confusion Matrix' + str(self.over_sampling) + str(self.method) +
            " " + str(self.learning_task) + ".png")
        plt.clf()

        # xgb.plot_tree(estimator)
        # plt.show()
        # xgb.plot_tree(estimator, num_trees=4)
        # plt.show()

    def model_evaluation(self, best_params, X_train=None, y_train=None, X_val=None, y_val=None):
        self.estimator = xgb.XGBClassifier(**best_params)
        if X_train is None & y_train is None & X_val is None & y_val is None:
            self.estimator.fit(self.X_train, self.y_train,
                               eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)])
        else:
            self.estimator.fit(X_train, y_train,
                               eval_set=[(X_train, y_train), (X_val, y_val)])

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
    df = pd.read_csv('6dof_disc_and_classification.csv')
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df = df.sample(frac=1, axis=1).reset_index(drop=True)
    data, name = learning_task(data=df, task='Binary')

    # print(data['class'].nunique())
    # results = pd.DataFrame(columns=['Learning Task', 'lb', 'ub', 'over sampling?', 'method'])
    # methods = [1, 2, 3]
    # over_sampling = [True, False]
    # learning_tasks = ['Binary']
    # lb = 3
    # ub = 9
    # for t in learning_tasks:
    #     data, learn_name = learning_task(data=df, task=t, lb=lb, ub=ub)
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
    #             xg = XGB(dataset=data, learning_task=learn_name, over_sampling=v, method=i, multiclass=multi,
    #                      num_class=n_class,
    #                      plot_path="C:\\Users\\azrie\\Desktop\\plots\\")
    #             best_params = xg.hyper_parameter_tuning()
    #             xg.model_evaluation(best_params)
    #             results.append({'Learning Task': t, 'lb': lb, 'ub': ub, 'over sampling?': v, 'method': i},
    #                            ignore_index=True)
    #
    # results.to_csv("Results_comparison.csv")
    # data = learning_task(data=df, task='3 class', lb=1, ub=9)
    # xg = XGB(dataset=data, over_sampling=True, method=1, multiclass=False,
    #          plot_path="C:\\Users\\azrie\\Desktop\\plots\\")
    # best_params = xg.hyper_parameter_tuning()
    # xg.plot_performance(best_params)
    # xg.model_evaluation(best_params)

    xg = XGB(dataset=data, learning_task='Binary', imbalanced=True, over_sampling=False, method=-1, multiclass=False,
             plot_path="C:\\Users\\azrie\\Desktop\\plots\\")
    xg.baseline()
    # best_params = xg.hyper_parameter_tuning()
    # xg.plot_performance(best_params)
    # xg.model_evaluation(best_params)
