# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:23:10 2021

@author: Rares
"""

import numpy as np
from librosa import feature

def extractFeatures(sound, sampleRate=44100, nCoeffs=32):
    """
    

    Parameters
    ----------
    sound : Numpy vector
        Contains the sound data.
    sampleRate : int, optional
        The sample rate in Hz. The default is 44100.
    nCoeffs : int, optional
        Number of MFCC coefficients. The default is 32.

    Returns
    -------
    mfccScaled : numpy array of shape (1, nCoeffs)
        The MFCC coefficients

    """
    mfcc = feature.mfcc(sound, sampleRate, n_mfcc=nCoeffs)
    mfccScaled = np.mean(mfcc.T, axis=0)
    return mfccScaled