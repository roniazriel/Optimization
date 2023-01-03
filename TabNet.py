import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetRegressor
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


class Tab_Net:
    def __init__(self, train_set, target1, target2, target3, targets):
        self.train_dataset = train_set
        self.target1 = target1
        self.target2 = target2
        self.target3 = target3
        self.targets = targets
        self.targets = targets
        self.split_data()

    def split_data(self):
        """
        :param train_df: dataframe of train dataset
        :param test_df: dataframe of validation dataset
        :param target1: target 1 column name as a string
        :param target2: target 2 column name as a string
        :param target3: target 3 column name as a string
        :return: x_test, y_test, x_val, y_val, x_test, y_test
        """
        train_df, test_df = train_test_split(self.train_dataset, test_size=0.2, random_state=200) # 80% of dataset for training, 20% of dataset as test
        # x_train, x_val, y_train, y_val = train_test_split(
        #     train_df.drop(columns=[self.target1, self.target2, self.target3]),
        #     train_df[[self.target1, self.target2, self.target3]],
        #     test_size=0.3)
        # x_test = test_df.drop(columns=[self.target1, self.target2, self.target3])
        # y_test = test_df[[self.target1, self.target2, self.target3]]

        x_train, x_val, y_train, y_val = train_test_split(train_df.drop(columns=[self.target1, self.target2, self.target3]),train_df[[self.target1, self.target2, self.target3]],test_size=0.3)
        x_test = test_df.drop(columns=[self.target1, self.target2, self.target3])
        y_test = test_df[[self.target1, self.target2, self.target3]]

        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_valid = x_val.to_numpy()
        y_valid = y_val.to_numpy()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()

        print("x train\n", x_train.shape)
        print("y train\n", y_train.shape)

        print("x valid\n", x_valid.shape)
        print("y valid\n", y_valid.shape)

        print("x test\n", x_test.shape)
        print("y test\n", y_test.shape)

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train, x_val, y_val, x_test, y_test

    def initial_model(self):
        clf1 = TabNetRegressor()
        # fit the model
        print(self.x_train)
        print(self.y_train)
        clf1.fit(
            self.x_train, self.y_train,
            eval_set=[(self.x_train, self.y_train), (self.x_val, self.y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['mae', 'mse'],
            max_epochs=1000, patience=50,
            batch_size=256, virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        preds = clf1.predict(self.x_test)

        test_mse = mean_squared_error(y_pred=preds, y_true=self.y_test)
        test_mae = mean_absolute_error(self.y_test, preds)

        print(f"BEST VALID SCORE: {clf1.best_cost}")
        print(f"MSE TEST SCORE: {test_mse}")
        print(f"MAE TEST SCORE: {test_mae}")
        return clf1

    def Objective(self, trial):
        """
        :param trial: Optuna study
        :return: CV average score
        """
        X = self.train_dataset.drop(columns=[self.target1,self.target2,self.target3]).values
        Y = self.train_dataset[[self.target1,self.target2,self.target3]].values
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 56, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
        tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                             lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                             mask_type=mask_type, n_shared=n_shared,
                             scheduler_params=dict(mode="min",
                                                   patience=trial.suggest_int("patienceScheduler", low=3, high=10),
                                                   # changing sheduler patience to be lower than early stopping patience
                                                   min_lr=1e-5,
                                                   factor=0.5, ),
                             scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                             verbose=0,
                             )  # early stopping
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        CV_score_array = []
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = Y[train_index], Y[test_index]
            regressor = TabNetRegressor(**tabnet_params)
            regressor.fit(X_train=X_train, y_train=y_train,
                          eval_set=[(X_valid, y_valid)],
                          patience=trial.suggest_int("patience", low=15, high=30),
                          max_epochs=trial.suggest_int('epochs', 1, 100),
                          eval_metric=['mse', 'mae'])
            CV_score_array.append(regressor.best_cost)
        avg = np.mean(CV_score_array)
        return avg

    def hyperparameter_tuning(self):
        study = optuna.create_study(direction="minimize", study_name='TabNet optimization')
        study.optimize(self.Objective, timeout=6 * 60)  # 5 hours
        TabNet_params = study.best_params

        final_params = dict(n_d=TabNet_params['n_da'], n_a=TabNet_params['n_da'], n_steps=TabNet_params['n_steps'],
                            gamma=TabNet_params['gamma'],
                            lambda_sparse=TabNet_params['lambda_sparse'], optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                            mask_type=TabNet_params['mask_type'], n_shared=TabNet_params['n_shared'],
                            scheduler_params=dict(mode="min",
                                                  patience=TabNet_params['patienceScheduler'],
                                                  min_lr=1e-5,
                                                  factor=0.5, ),
                            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                            verbose=0,
                            )
        epochs = TabNet_params['epochs']

        regressor = TabNetRegressor(**final_params)
        # regressor.fit(self.x_train, self.y_train,
        #               eval_set=[(self.x_train, self.y_train), (self.x_val, self.y_val)],
        #               eval_name=['train', 'valid'],
        #               patience=TabNet_params['patience'], max_epochs=epochs,
        #               eval_metric=['mse', 'mae'])
        return regressor


    def analysis(self, regressor):
        regressor.fit(self.x_train,self.y_train,
            eval_set=[(self.x_train, self.y_train), (self.x_valid, self.y_valid)],
            eval_name=['train', 'valid'],
            max_epochs=1000,
            eval_metric=['mse','mae'])

        preds = regressor.predict(self.x_test) ### Predictions on train set

        manipulability_preds = [row[0] for row in preds]
        success_preds = [row[1] for row in preds]
        manipulability_true = [row[0] for row in self.y_test]
        success_true = [row[1] for row in self.y_test]
        preds_true_df = pd.DataFrame({'MANIPULABILITY PREDS': manipulability_preds,'SUCSSES RATE PREDS': success_preds, 'MANIPULABILITY TRUE VALUE': manipulability_true,'SUCSSES RATE TRUE VALUE': success_true}, columns=['MANIPULABILITY PREDS','SUCSSES RATE PREDS' ,'MANIPULABILITY TRUE VALUE','SUCSSES RATE TRUE VALUE'])

        test_acc1 = r2_score(manipulability_true,manipulability_preds)
        test_acc2 = r2_score( success_true,success_preds)

        test_mse = mean_squared_error(y_pred=preds, y_true=self.y_test)
        test_mae = mean_absolute_error(y_pred=preds, y_true=self.y_test)

        print(f"BEST VALID SCORE: {regressor.best_cost}")
        print(f"MSE TEST SCORE: {test_mse}")
        print(f"MAE TEST SCORE: {test_mae}")
        print(f"R SQURE FOR MANIPULABILITY TEST SCORE: {test_acc1}")
        print(f"R SQURE FOR SUCSSES RATE TEST SCORE: {test_acc2}")

        preds_true_df.to_csv('C:/Users/azrie/PycharmProjects/pythonProject/DL/predictions.csv')

        '''Manipulability scores'''
        yhat = np.array(manipulability_preds)
        SS_Residual = sum((np.array(manipulability_true)-yhat)**2)
        SS_Total = sum((manipulability_true-np.mean(manipulability_true))**2)
        r_squared = 1 - (float(SS_Residual))/SS_Total
        adjusted_r_squared = 1 - (1-r_squared)*(len(self.y_test)-1)/(len(manipulability_true)-self.x_test.shape[1]-1)
        print ('r_squared for manipulability:',r_squared)
        print('adjusted_r_squared for manipulability:', adjusted_r_squared)

        '''Succses index scores'''
        yhat = np.array(success_preds)
        SS_Residual = sum((np.array(success_true)-yhat)**2)
        SS_Total = sum((success_true-np.mean(success_true))**2)
        r_squared = 1 - (float(SS_Residual))/SS_Total
        adjusted_r_squared = 1 - (1-r_squared)*(len(self.y_test)-1)/(len(success_true)-self.x_test.shape[1]-1)
        print ('r_squared for Succses index:',r_squared)
        print('adjusted_r_squared for Succses index:', adjusted_r_squared)

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

        '''Residuals V.S Predictions'''
        residuals = np.array(manipulability_true) - np.array(manipulability_preds)
        plt.scatter(residuals,manipulability_preds)
        plt.xlabel("Residuals")
        plt.ylabel("True Manipulability")
        plt.title("Manipulability- TRUE V.S PREDICTED")
        plt.show()

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

        '''change reability predictions to the closest digit'''
        rounded_reachability = np.round(success_preds,1)
        print(rounded_reachability)
        print(success_preds)

        '''Residuals V.S Predictions'''
        residuals = np.array(success_true) - np.array(rounded_reachability)
        plt.scatter(residuals,rounded_reachability)
        plt.xlabel("Residuals")
        plt.ylabel("True Sucsses")
        plt.title("Sucsses- TRUE V.S PREDICTED")
        plt.show()

        q25, q75 = np.percentile(residuals, [25, 75])
        bin_width = 2 * (q75 - q25) * len(residuals) ** (-1/3)
        bins = round((residuals.max() - residuals.min()) / bin_width)
        print("Freedman–Diaconis number of bins:", bins)
        plt.hist(residuals, bins=bins)
        plt.title('Histogram of Success Residuals')
        plt.ylabel('Count')
        plt.xlabel('residuals');

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

        '''Covariance ot the two outputs - indicates the relationship of two variables'''
        # grouped_data[['Manipulability_Rates', 'Success_Rates']].cov()
        #
        #
        # '''Variance ot the two outputs - Manipulability_Rates is normalized'''
        # grouped_data[['Manipulability_Rates', 'Success_Rates']].std()


if __name__ == '__main__':
    train_df = pd.read_csv('6dof_grouped_dummied.csv')
    train_df = train_df.drop(columns=['Unnamed: 0','Max_Manipulability','Manipulability_Rates'])
    print(train_df)
    target1 = 'Min_Manipulability'
    target2 = 'Success_Rates'
    target3 = 'MaxSum_Mid_joint_proximity'
    targets = [target1,target2,target3]

    tab = Tab_Net(train_df, target1, target2, target3,targets)
    tab.initial_model()
    # best_tab = tab.hyperparameter_tuning()
    # path = 'TabNet_Model.zip'
    # torch.save(best_tab, path)
    # model = torch.load(path)
    # print(model)

    # saved_filename = best_tab.save_model(path)
    # # define new model and load save parameters
    # tabnet_regressor = TabNetRegressor()
    # tabnet_regressor.load_model('TabModel.zip')

