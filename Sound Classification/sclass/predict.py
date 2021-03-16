# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 02:13:16 2021

@author: Rares
"""

import xgboost as xgb
from .preprocess import extractFeatures
import librosa
from librosa import util

def predictSoundClass(sound, boostModel, sampleRate=44100, nCoeffs=32):
    """
    Predicts the sound class (0 -> Kick, 1 -> Snare) for a single sound
    using an XGBoost model

    Parameters
    ----------
    sound : Numpy vector
        Sound data
    boostModel : XGB Booster class
        The XGBoost model
    sampleRate : int, optional
        Sound sample rate in Hz. The default is 44100.
    nCoeffs : int, optional
        Number of MFCC coefficients (features). The default is 32.

    Returns
    -------
    Float between [0, 1] - class probability
        0 -> Kick, 1 -> Snare

    """
    sound = util.normalize(sound)
    mfcc = extractFeatures(sound, sampleRate, nCoeffs)
    mfcc = mfcc.reshape(1, len(mfcc))
    dTest = xgb.DMatrix(mfcc)
    return boostModel.predict(dTest)