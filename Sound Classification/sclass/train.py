# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:23:10 2021

@author: Rares
"""
import xgboost as xgb

def trainModel(trainX, testX, trainY, testY, modelPath='Binary Model'):
    """
    Trains the classification model and saves it to modelPath

    Parameters
    ----------
    trainX : pandas DataFrame
        Train features
    testX : pandas DataFrame
        Test features
    trainY : pandas DataFrame
        Train labels
    testY : pandas DataFrame
        Test labels
    modelPath : String, optional
        Path for saving the trained model (do not include the file termination). 
        The default is 'Binary Model'.

    """
    param = {'eval_metric': 'error',
             'objective': 'binary:logistic'}
    nRounds = 500
    dTrain = xgb.DMatrix(trainX, trainY)
    dEval = xgb.DMatrix(testX, testY)
    
    evalList = [(dEval, 'eval'), (dTrain, 'train')]
    boostTrainer = xgb.train(param, dTrain, nRounds, evalList, early_stopping_rounds=5, verbose_eval=1)
    boostTrainer.save_model(modelPath + '.model')
    
