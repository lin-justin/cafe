#!/usr/bin/env python3

# Author: Justin Lin, M.S.

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
np.random.seed(9999)
import pandas as pd 
from collections import Counter

from scikitplot.metrics import plot_roc
import matplotlib.pyplot as plt 
import seaborn as sns

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as imbl_pipeline
import eli5

from sklearn.pipeline import Pipeline as skl_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import matthews_corrcoef, classification_report, log_loss, plot_roc_curve, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

import random
random.seed(9999)
from time import time

#############################
#      Classification       #
#############################

def classify(features, labels, model = 'all', resample_method = None, scoring = 'roc_auc_ovo', cv = 10, n_iter = 10):
    '''
    A nested function to apply machine learning classification

    Args:
        features: A pandas dataframe containing the features
        labels: A pandas dataframe containing the labels
        model: Options are: 'rf' - Random Forest
                            'gbm' - Gradient Boosting
                            'dt' - Decision Tree
                            'et' - Extremely Randomized Tree 
                            'log_sgd' - Logistic Regression with Stochastic
                                        Gradient Descent learning
                            'all' - Tests out all five of the models and identifies
                                    which model is the best based on the cross-validation
                                    score
                Default is 'all'
        resample_method: Resampling to deal with imbalanced data
                         Reference: https://imbalanced-learn.readthedocs.io/en/stable/combine.html#bpm2004
                         Options are:
                                    'smote_tomek' - https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTETomek.html#imblearn.combine.SMOTETomek
                                    'smote_enn' - https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTEENN.html#imblearn.combine.SMOTEENN
                         Default is None
        scoring: The metric for evaluating model performance
                 Reference: https://scikit-learn.org/stable/modules/model_evaluation.html
                 Default is 'roc_auc_ovo'
        cv: The number of splits for cross-validation
            Default is 10
        n_iter: The number of parameter settings that are sampled
                Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
                Default is 10

    Returns
            the tuned classifier
            a report containing the f1-score, precision, and recall for each class
            the Matthews Correlation Coefficient value
            the log loss (cross-entropy loss)
            a pandas dataframe of the features predicted to be of a certain class given its weight and value
            a confusion matrix figure
            a ROC curve figure
    '''

    # Make sure the features and labels are of type pandas dataframes
    if not isinstance(features, pd.DataFrame) and not isinstance(labels, pd.DataFrame):
        raise TypeError('The features and labels are not of a pandas dataframe type.')

    # Make sure the number of rows are the same in features and labels
    assert features.shape[0] == labels.shape[0], 'Unequal number of rows.'

    # Get the names of the features
    feature_names = features.columns.values

    def standardize(X):
        '''
        Custom standardization function

        Args:
            X: the features as a numpy array

        Returns the standardized features
        '''
        return (X - np.mean(X))/np.std(X)

    def which_model(X, y, model = 'all'):
        '''
        Using the baseline models (default parameters) of Random Forest, 
        Gradient Boosting, Decision Tree, Extremely Randomized Tree, and 
        Logistic Regression with Stochastic Gradient Descent learning on 
        the entire dataset (with cross-validation) to determine which model is best.
        
        The user can either test the 5 models individually or test all of them by
        setting model = 'all'
        
        Args:
            X: The numpy array containing the features
            y: The numpy array containing the labels
            model: Options are: 'rf' - Random Forest
                                'gbm' - Gradient Boosting
                                'dt' - Decision Tree
                                'et' - Extremely Randomized Tree 
                                'log_sgd' - Logistic Regression with Stochastic
                                            Gradient Descent learning
                                'all' - Tests out all five of the models and identifies
                                        which model is the best based on the cross-validation
                                        score
                    Default is 'all'
                
        Returns best model
        '''

        if model == 'all':
            # Pipelines help prevent data leakage
            pipelines = []
            pipelines.append(('Random Forest', skl_pipeline([('Standardization', FunctionTransformer(standardize)),
                                                        ('RF', RandomForestClassifier(random_state = 9999))])))
            pipelines.append(('Gradient Boosting', skl_pipeline([('Standardization', FunctionTransformer(standardize)),
                                                        ('GBM', GradientBoostingClassifier(random_state = 9999))])))
            pipelines.append(('Decision Tree', skl_pipeline([('Standardization', FunctionTransformer(standardize)),
                                                        ('DT', DecisionTreeClassifier(random_state = 9999))])))
            pipelines.append(('Extra Trees', skl_pipeline([('Standardization', FunctionTransformer(standardize)),
                                                        ('ET', ExtraTreesClassifier(random_state = 9999))])))
            pipelines.append(('Logistic Regression (SGD)', skl_pipeline([('Standardization', FunctionTransformer(standardize)),
                                                        ('LOGSGD', SGDClassifier(loss = 'log', random_state = 9999))])))

            print('\nSelecting model...')
            print('\nModel\tScore')
            print('-------------')
            results = []
            names = []
            for name, model in pipelines:
                # Apply cross validation
                cv_results = cross_val_score(model, X, y, cv = cv, scoring = scoring)
                results.append(np.mean(cv_results))
                names.append(name)
                print('{}: {:.4f} ± {:.4f}'.format(name, np.mean(cv_results), np.std(cv_results)))
            
            names_results = list(zip(names, results))
            # Return model with highest score value
            selected_model = max(names_results, key = lambda item:item[1])
            print('\nThe selected model is', selected_model)
            
            if 'Gradient Boosting' in selected_model:
                return GradientBoostingClassifier()
            elif 'Random Forest' in selected_model:
                return RandomForestClassifier()
            elif 'Decision Tree' in selected_model:
                return DecisionTreeClassifier()
            elif 'Extra Trees' in selected_model:
                return ExtraTreesClassifier()
            elif 'Logistic Regression (SGD)' in selected_model:
                return SGDClassifier(loss = 'log')

        elif model == 'rf':
            rf_pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('RF', RandomForestClassifier(random_state = 9999))])
            rf_score = cross_val_score(rf_pipe, X, y, cv = cv, scoring = scoring)
            print('\nRandom Forest score: {:.4f} ± {:.4f}'.format(np.mean(rf_score), np.std(rf_score)))
            return RandomForestClassifier()

        elif model == 'gbm':
            gb_pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('GBM', GradientBoostingClassifier(random_state = 9999))])
            gb_score = cross_val_score(gb_pipe, X, y, cv = cv, scoring = scoring)
            print('\nGradient Boosting score: {:.4f} ± {:.4f}'.format(np.mean(gb_score), np.std(gb_score)))
            return GradientBoostingClassifier()

        elif model == 'dt':
            dt_pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('DT', DecisionTreeClassifier(random_state = 9999))])
            dt_score = cross_val_score(dt_pipe, X, y, cv = cv, scoring = scoring)
            print('\nDecision Tree score: {:.4f} ± {:.4f}'.format(np.mean(dt_score), np.std(dt_score)))
            return DecisionTreeClassifier()

        elif model == 'et':
            et_pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('ET', ExtraTreesClassifier(random_state = 9999))])
            et_score = cross_val_score(et_pipe, X, y, cv = cv, scoring = scoring)
            print('\nExtra Trees score: {:.4f} ± {:.4f}'.format(np.mean(et_score), np.std(et_score)))
            return ExtraTreesClassifier()
        
        elif model == 'log_sgd':
            log_pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('LOGSGD', SGDClassifier(loss = 'log', random_state = 9999))])
            log_score = cross_val_score(log_pipe, X, y, cv = cv, scoring = scoring)
            print('\nLogistic Regression (SGD) score: {:.4f} ± {:.4f}'.format(np.mean(log_score), np.std(log_score)))
            return SGDClassifier(loss = 'log')

    def train(selected_model, X_train, y_train, resample_method = None):
        '''
        Train and tune the hyperparameters of the selected model

        Random Search is used because the parameter search space is large
        and performs as well as Grid Search.

        Hyperparameter tuning is more art than science as it is based on
        expertise and experience

        Args:
            selected_model: The model from which_model()
            X_train: The training features
            y_train: The training labels

        Returns the tuned classifier
        '''
        
        print('\nStarting training...')

        if resample_method == None:

            if selected_model.__class__.__name__ == 'GradientBoostingClassifier':
                
                start = time()

                pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('clf', selected_model)])
                
                gb_grid = {'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
                        'clf__subsample': [0.7, 0.8],
                        'clf__learning_rate': [0.001, 0.01, 0.1],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'clf__max_features': ['sqrt', 'log2'],
                        'clf__min_samples_split':  [2, 5, 10],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__loss': ['deviance'],
                        'clf__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe, 
                                        gb_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf
            
            elif selected_model.__class__.__name__ == 'RandomForestClassifier':
                
                start = time()

                pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('clf', selected_model)])
                
                rf_grid = {'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__bootstrap': [True],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe,
                                        rf_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf

            elif selected_model.__class__.__name__ == 'DecisionTreeClassifier':
                
                start = time()
                
                pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('clf', selected_model)])
                
                dt_grid = {'clf__criterion': ['gini'],
                        'clf__splitter': ['best', 'random'],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe,
                                        dt_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf

            elif selected_model.__class__.__name__ == 'ExtraTreesClassifier':
                
                start = time()

                pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('clf', selected_model)])

                et_grid = {'clf__criterion': ['gini'],
                        'clf__bootstrap': [True, False],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999]}

                clf = RandomizedSearchCV(pipe,
                                        et_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)

                clf.fit(X_train, y_train)

                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))

                return clf

            elif 'log' in selected_model.get_params().values():

                start = time()

                pipe = skl_pipeline([('Standardization', FunctionTransformer(standardize)), ('clf', selected_model)])

                log_grid = {'clf__loss': ['log'],
                            'clf__penalty': ['l2', 'l1', 'elasticnet'],
                            'clf__alpha': [0.01, 0.001, 0.0001],
                            'clf__max_iter': [1000, 5000],
                            'clf__class_weight': ['balanced', None],
                            'clf__random_state': [9999]}

                clf = RandomizedSearchCV(pipe,
                                        log_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)

                clf.fit(X_train, y_train)

                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))

                return clf

        elif resample_method == 'smote_tomek':

            if selected_model.__class__.__name__ == 'GradientBoostingClassifier':
                
                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTETOMEK', SMOTETomek()), ('clf', selected_model)])
                
                gb_grid = {'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
                        'clf__subsample': [0.7, 0.8],
                        'clf__learning_rate': [0.001, 0.01, 0.1],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'clf__max_features': ['sqrt', 'log2'],
                        'clf__min_samples_split':  [2, 5, 10],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__loss': ['deviance'],
                        'clf__random_state': [9999],
                        'SMOTETOMEK__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe, 
                                        gb_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf
            
            elif selected_model.__class__.__name__ == 'RandomForestClassifier':
                
                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTETOMEK', SMOTETomek()), ('clf', selected_model)])
                
                rf_grid = {'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__bootstrap': [True],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999],
                        'SMOTETOMEK__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe,
                                        rf_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf

            elif selected_model.__class__.__name__ == 'DecisionTreeClassifier':
                
                start = time()
                
                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTETOMEK', SMOTETomek()), ('clf', selected_model)])
                
                dt_grid = {'clf__criterion': ['gini'],
                        'clf__splitter': ['best', 'random'],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999],
                        'SMOTETOMEK__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe,
                                        dt_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf

            elif selected_model.__class__.__name__ == 'ExtraTreesClassifier':
                
                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTETOMEK', SMOTETomek()), ('clf', selected_model)])

                et_grid = {'clf__criterion': ['gini'],
                        'clf__bootstrap': [True, False],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999]}

                clf = RandomizedSearchCV(pipe,
                                        et_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)

                clf.fit(X_train, y_train)

                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))

                return clf

            elif 'log' in selected_model.get_params().values():

                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTETOMEK', SMOTETomek()), ('clf', selected_model)])

                log_grid = {'clf__base_estimator__loss': ['log'],
                            'clf__base_estimator__penalty': ['l2', 'l1', 'elasticnet'],
                            'clf__base_estimator__alpha': [0.01, 0.001, 0.0001],
                            'clf__base_estimator__max_iter': [1000, 5000],
                            'clf__base_estimator__class_weight': ['balanced', None],
                            'clf__base_estimator__random_state': [9999],
                            'SMOTETOMEK__random_state': [9999]}

                clf = RandomizedSearchCV(pipe,
                                        log_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)

                clf.fit(X_train, y_train)

                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))

                return clf

        elif resample_method == 'smote_enn':

            if selected_model.__class__.__name__ == 'GradientBoostingClassifier':
                
                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTENN', SMOTEENN()), ('clf', selected_model)])
                
                gb_grid = {'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
                        'clf__subsample': [0.7, 0.8],
                        'clf__learning_rate': [0.001, 0.01, 0.1],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                        'clf__max_features': ['sqrt', 'log2'],
                        'clf__min_samples_split':  [2, 5, 10],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__loss': ['deviance'],
                        'clf__random_state': [9999],
                        'SMOTENN__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe, 
                                        gb_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf
            
            elif selected_model.__class__.__name__ == 'RandomForestClassifier':
                
                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTENN', SMOTEENN()), ('clf', selected_model)])
                
                rf_grid = {'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__bootstrap': [True],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999],
                        'SMOTENN__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe,
                                        rf_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf

            elif selected_model.__class__.__name__ == 'DecisionTreeClassifier':
                
                start = time()
                
                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTENN', SMOTEENN()), ('clf', selected_model)])
                
                dt_grid = {'clf__criterion': ['gini'],
                        'clf__splitter': ['best', 'random'],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999],
                        'SMOTENN__random_state': [9999]}
                
                clf = RandomizedSearchCV(pipe,
                                        dt_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)
                
                clf.fit(X_train, y_train)
                
                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))
                
                return clf

            elif selected_model.__class__.__name__ == 'ExtraTreesClassifier':
                
                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTENN', SMOTEENN()), ('clf', selected_model)])

                et_grid = {'clf__criterion': ['gini'],
                        'clf__bootstrap': [True, False],
                        'clf__max_depth': [int(x) for x in np.linspace(10, 110, num= 11)],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__max_features': ['sqrt', 'auto'],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__class_weight': ['balanced', None],
                        'clf__random_state': [9999]}

                clf = RandomizedSearchCV(pipe,
                                        et_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)

                clf.fit(X_train, y_train)

                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))

                return clf

            elif 'log' in selected_model.get_params().values():

                start = time()

                pipe = imbl_pipeline([('Standardization', FunctionTransformer(standardize)), ('SMOTENN', SMOTEENN()), ('clf', selected_model)])

                log_grid = {'clf__loss': ['log'],
                            'clf__penalty': ['l2', 'l1', 'elasticnet'],
                            'clf__alpha': [0.01, 0.001, 0.0001],
                            'clf__max_iter': [1000, 5000],
                            'clf__class_weight': ['balanced', None],
                            'clf__random_state': [9999],
                            'SMOTENN__random_state': [9999]}

                clf = RandomizedSearchCV(pipe,
                                        log_grid,
                                        cv = cv,
                                        n_iter = n_iter,
                                        scoring = scoring,
                                        n_jobs = -1,
                                        random_state = 9999)

                clf.fit(X_train, y_train)

                end = time()

                time_elapsed = end - start
                
                print('\nThe best {}-fold cross valdiation score is {:.4f}.'.format(cv, clf.best_score_))
                print('The best parameters are:\n', clf.best_estimator_.get_params()['clf'])
                print('Training took {:.0f}m {:.0f}s.'.format(time_elapsed//60, time_elapsed % 60))

                return clf

    def evaluate(clf, X_test, y_test):
        '''
        Evaluate the tuned classifier's performance on the testing set

        Args:
            clf: The tuned classifier from train()
            X_test: The test data features
            y_test: The test data labels

        Returns 
                a report containing the f1-score, precision, and recall of each class
                the Matthew's Correlation Coefficient
                the log loss (cross-entropy loss)
                a confusion matrix figure
                a ROC curve figure
        '''

        def plot_confusion_matrix(y_test, y_pred):
            '''
            Confusion matrix

            Args:
                y_test: The test set labels
                y_pred: The predicted labels

            Returns confusion matrix figure
            '''
            cm = confusion_matrix(y_test, y_pred)
            cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]

            df_cm = pd.DataFrame(cm, columns = np.unique(y_test),
                                index = np.unique(y_test))
            df_cm.index.name = 'True Labels'
            df_cm.columns.name = 'Predicted Labels'

            cm_fig = sns.heatmap(df_cm, cmap = 'Blues', 
                                 annot = True, 
                                 cbar = False)
            for _, spine in cm_fig.spines.items():
                spine.set_visible(True)

            plt.title('{} Confusion Matrix'.format(model_name))
            plt.yticks(rotation = 0)

            return cm_fig

        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)

        mcc = matthews_corrcoef(y_test, y_pred)

        if 'log' in clf.best_estimator_.get_params().values():
            model_name = 'Logistic Regression (SGD)'
            y_probas = clf.best_estimator_['clf'].predict_proba(X_test)
        else:
            model_name = selected_model.__class__.__name__
            y_probas = clf.predict_proba(X_test)

        conf_mat = plot_confusion_matrix(y_test, y_pred)

        # Binary classification vs multi-class classification ROC curve
        if len(np.unique(y_test)) > 2:
            roc_curve = plot_roc(y_test, y_probas, title = '{} ROC curve'.format(model_name))
        else:
            roc_curve = plot_roc_curve(clf, X_test, y_test, name = model_name)
            roc_curve.figure_.suptitle('{} ROC curve'.format(model_name))

        loss_score = log_loss(y_test, y_probas)

        # Get the features of their predicted class based on weight and value
        if 'log' in selected_model.get_params().values():
            feat_imp = eli5.sklearn.explain_prediction_linear_classifier(clf.best_estimator_['clf'], X_test[1], feature_names = feature_names)
        else:
            feat_imp = eli5.sklearn.explain_prediction.explain_prediction_tree_classifier(clf.best_estimator_['clf'], X_test[1], feature_names = feature_names)

        return report, mcc, loss_score, feat_imp, conf_mat, roc_curve

    
    X = features.to_numpy()
    y = labels.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 9999)

    selected_model = which_model(X, y, model = model)

    tuned_clf = train(selected_model, X_train, y_train, resample_method = resample_method)

    report, mcc, loss_score, feat_imp, conf_mat, roc_curve = evaluate(tuned_clf, X_test, y_test)

    feat_imp = eli5.format_as_dataframe(feat_imp)

    return tuned_clf, report, mcc, loss_score, feat_imp, conf_mat, roc_curve