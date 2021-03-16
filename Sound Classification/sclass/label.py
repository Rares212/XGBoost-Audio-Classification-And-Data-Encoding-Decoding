# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:23:10 2021

@author: Rares
"""
from .preprocess import extractFeatures
import librosa
from librosa import util
import xgboost as xgb
import pandas as pd
import numpy as np
from object_cache import object_cache

@object_cache
def loadAndGetMfcc(folder, sampleRate=44100, nCoeffs=32):
    """
    

    Parameters
    ----------
    folder : String
        Path to the folder containing the samples
    sampleRate : int, optional
        Sample rate in Hz
        The default is 44100.
    nCoeffs : int, optional
        Number of MFCC coefficients to obtain
        The default is 32.

    Returns
    -------
    mfccCoeffs : Numpy array of shape ([number of files], nCoeffs)
        The MFCC coefficients for each audio file

    """
    files = util.find_files(folder, recurse=False)
    nFiles = len(files)
    mfccCoeffs = np.empty((nFiles, nCoeffs))
    fileIndex = 0
    for filename in files:
        (sound, fs) = librosa.load(filename, sampleRate, res_type="kaiser_fast")
        sound = util.normalize(sound)
        mfccCoeffs[fileIndex] = extractFeatures(sound, sampleRate, nCoeffs)
        fileIndex += 1
    return mfccCoeffs

def combineFeatures(kickFeatures, snareFeatures):
    """
    Gets the training feature and label arrays from the kick and snare features
    Kick -> 0
    Snare -> 1

    Parameters
    ----------
    kickFeatures : Numpy array of shape ([number of sounds], [number of coefficients])
        MFCC feature array for kicks
    snareFeatures : Numpy array of shape ([number of sounds], [number of coefficients])
        MFCC feature array for snares

    Returns
    -------
    features: numpy array containing all features
    labels: numpy array containing the feature labels

    """
    nKicks = kickFeatures.shape[0]
    nSnares = snareFeatures.shape[0]
    
    features = np.concatenate((kickFeatures, snareFeatures))
    labels = np.empty((nKicks + nSnares, 1))
    labels[0:nKicks] = 0
    labels[nKicks:] = 1
    #featuresFrame = pd.DataFrame(features)
    #labelFrame = pd.DataFrame(labels)
    #return (featuresFrame, labelFrame)
    
    return (features, labels)
        