import pandas as pd

import numpy as np

import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 

from scipy.stats import uniform, norm



#### Functions ####

def profile_clf(X,y,clf,X_val = None, y_val = None,cv=5,train_scorer = 'accuracy',scoring = {'Accuracy': 'accuracy'}):

    '''
    Prints out the score for a model on the training set and the validation set 
    for a number of metrics, i.e. a 'profile' of the model. If no validation set is given then it uses cross validation.

    Parameters
    ----------
    X: numpy array or pandas dataframe.
        The training set.
    y: numpy array or pandas dataframe.
        The training target variable.
    X_val: numpy array or pandas dataframe.
        The validation set.
    y_val: numpy array or pandas dataframe.
        The validation target variable.
    clf: a scikit-learn model. 
        The classifier to get a profile of.
    cv: int. 
        The number of folds to use in cross-validation.
    train_scorer: string, corresponding to a scikit-learn metric.
        The metric to measure performance on the training set.
    scoring: dict of (string, string) pairs, where the second string corresponds to a scikit-learn metric.
        The metric(s) to us eon cross validation.
    '''
    
    scorer = sklearn.metrics.get_scorer(train_scorer)
    print(train_scorer, 'on the training set :', scorer(clf, X, y))
    
    if X_val is not None:
        for score in scoring.keys():
            scorer = sklearn.metrics.get_scorer(scoring[score])
            print(score, ' on validation set:', scorer(clf, X_val,y_val))
    
    else:
        scores = cross_validate(clf,X,y, cv=cv, scoring = scoring, n_jobs = -1)
        for score in scoring.keys():
            print(score, ' calculated via %d-fold cross-validation:%0.3f(+/-%0.3f)'
                    %(cv,scores[f'test_{score}'].mean(),scores[f'test_{score}'].std()*2))    
    
def grid_search_cv(X,y,clf, param_grid,cv = 5, return_train_score = True, n_jobs = -1, **kwargs): 

    '''
    Performs grid search, best model, returns the best model, and prints out its parameters.
    
    Parameters
    ----------
    X: numpy array or pandas dataframe.
        Dataset on which the model has been fit.
    y: numpy array or pandas dataframe.
        Target variable on which the model has been fit.
    
    **kwargs: parameters for GridSearchCV.
        By default has cv =5, return_train_score = True, and n_jobs = -1.
    '''
    
    grid_search = GridSearchCV(clf, param_grid,cv = cv, 
                    return_train_score = return_train_score, n_jobs = n_jobs, 
                    **kwargs)
                    
    grid_search.fit(X,y)
    print('Best paramaters:', grid_search.best_params_)
    return grid_search.best_estimator_

def random_search_cv(X,y,clf, param_grid,cv = 5, n_iter = 50, n_jobs = -1, 
                    return_train_score = True, **kwargs):
                    
                    
    '''
    Performs grid search, best model, returns the best model, and prints out its parameters.
    
    Parameters
    ----------
    X: numpy array or pandas dataframe.
        Dataset on which the model has been fit.
    y: numpy array or pandas dataframe.
        Target variable on which the model has been fit.
    
    **kwargs: parameters for RandomizedSearchCV.
        By default has cv =5, return_train_score = True, n_jobs = -1, n_iter = 50.
        
    '''
    
    rand_search = RandomizedSearchCV(clf,param_grid, cv = cv, return_train_score = return_train_score, 
                            n_jobs = -1, n_iter = n_iter,
                            **kwargs)
                            
    rand_search.fit(X,y)
    print('Best paramaters:', rand_search.best_params_)
    return rand_search.best_estimator_

def optimise_log_reg_params(X,y, cv = 4, scoring = None, n_iter = 100):

    '''
    Optimises a logistic regression classifier using random_search_cv().
    '''

    param_grid = [{'C': uniform(0,1),
              'solver':['newton-cg','lbfgs','liblinear']
                },
                # {'C': uniform(0,1),
                # 'penalty': ['elasticnet'],
                # 'l1_ratio': uniform(0,1)
                # }
                ]
     
    log_reg_clf = random_search_cv(X, y, clf =LogisticRegression(max_iter = 1000), 
                             cv = cv, param_grid = param_grid,n_iter = n_iter,
                             scoring = scoring)
                             
    return log_reg_clf 
        
def optimise_svm_params(X,y, cv = 4, scoring = None):

    '''
    Optimises an SVM classifier using random_search_cv().
    '''

    param_grid = {'C': norm(loc = 1.0,scale = 0.3),
               'gamma': uniform(0, 2),
              'kernel':['sigmoid', 'linear', 'rbf','poly'],
              'degree': range(1, 15)
              }

    svm_clf = random_search_cv(X,y,param_grid = param_grid, cv = cv, 
                           clf = SVC(probability = True), 
                           n_iter = 100,scoring = scoring)
    return svm_clf
    
def optimise_svr_params(X,y, cv = 5, scoring = None, n_iter = 50):

    '''
    Optimises an SVM classifier using random_search_cv().
    '''

    param_grid = {'C': norm(loc = 1.0,scale = 0.3),
                'epsilon': uniform(0.01, 0.5),
              'kernel':['sigmoid', 'rbf'],
              'degree': range(3, 10)
              }

    svm_clf = random_search_cv(X,y,clf = SVR(),
                            param_grid = param_grid, cv = cv, 
                            n_iter = n_iter,scoring = scoring)
    return svm_clf
   

def optimise_knn_params(X,y , cv = 5, scoring = None, n_neighbours = range(1, 10),
                        n_iter = 50, 
                        weights = ['uniform','distance'],p = [1,2,3],**kwargs):

    '''
    Optimises a k-nearest neighbours classifier using random_search_cv().
    '''
    
    param_grid = {'n_neighbors': n_neighbours,
              'weights': weights,
              'p': p,
              }

    knn_clf = grid_search_cv(X, y, param_grid = param_grid, 
                         cv = cv, clf = KNeighborsClassifier(**kwargs),
                         scoring = scoring)
                         
    return knn_clf 
    
def optimise_random_forest_params(X,y , cv = 5, 
                                    scoring = None,max_features = range(1,10),
                                    min_samples_leaf = [3,4,5,6,7,8,9,10],max_depth = [None],n_estimators = [100],**kwargs):
                                    
    '''
    Optimises a random forest classifier using random_search_cv().
    '''

    param_grid = {
              'n_estimators': n_estimators,
              'max_features': max_features,
              'min_samples_leaf': min_samples_leaf,
              'max_depth': max_depth                         
              }

    random_forest_clf = grid_search_cv(X, y, 
                        param_grid = param_grid, cv = cv,
                        clf = RandomForestClassifier(**kwargs), scoring = scoring)
                        
    return random_forest_clf
 
def optimise_gaussian_process_params(X,y , cv = 4, 
                                        scoring = None): 
                                        
    '''
    Optimises a Gaussian process classifier using grid_search_cv().
    '''

    param_grid = {'kernel': [1.0*RBF(1.0), Matern()],
              }

    gaussian_process_clf = grid_search_cv(X, y, 
                            param_grid = param_grid, cv = cv, scoring = scoring,
                            clf = GaussianProcessClassifier(max_iter_predict = 200, 
                                                            n_restarts_optimizer = 10))
                        
    return gaussian_process_clf

def get_relevant_features(X,y,clf, feature_importances, 
                            scoring = None, cv = 5, min_features = 3):

    '''
    Determines the number n such that the classifier performs
    optimally when given the first n features sorted by feature 
    importance using RedundantAttributesRemover. 
    
    min_features is the minimum number of features to consider. Setting 
    this below the parameter max_features for a RandomForestClassifier will 
    give an error.
    '''
    feature_selection_pipeline = Pipeline([
            ('feature_selection', RedundantAttributesRemover(feature_importances)),
            ('clf', LogisticRegression())
        ])

    param_grid = [{'feature_selection__n_features':  list(range(min_features,len(feature_importances)+1)),
                  'clf': [clf]
                 }]
    
    feature_selector = GridSearchCV(feature_selection_pipeline,param_grid, cv = cv, 
                   return_train_score = True, n_jobs = -1, scoring = scoring)
    feature_selector.fit(X,y)
    n_features = feature_selector.best_params_['feature_selection__n_features']
    
    return n_features

#### Transformers ####

class RedundantAttributesRemover(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, n_features = 5):
        self.feature_importances = feature_importances
        self.n_features = n_features
    def fit(self, X, y=None):
        sorted_importances = sorted(self.feature_importances, reverse = True)
        stored_importances = sorted_importances[:self.n_features]
        self.stored_indices = [list(self.feature_importances).index(i) for i in stored_importances]
        return self
    def transform(self, X):
        return X[:, self.stored_indices]


