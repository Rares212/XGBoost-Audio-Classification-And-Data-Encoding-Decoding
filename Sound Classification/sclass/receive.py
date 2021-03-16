# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:29:14 2021

@author: Rares
"""

from librosa import onset, stft
from .predict import predictSoundClass
import matplotlib.pyplot as plt
from librosa import util, display, effects, feature
import numpy as np

def decodeBitstreamFromAudio(audioStream, model, sampleRate=44100, detectionThreshold=0.3):
    """
    

    Parameters
    ----------
    audioStream : numpy vector
        The encoded audio stream
    model : Booster object
        The XGBoost model used for prediction
    sampleRate : int, optional
        The sample rate in Hz. The default is 44100.
    detectionThreshold : float, optional
        Between [0, 0.5]
        If the predictor returns a value between [0, detectionThreshold]
        or [detectionThreshold, 1] the bit will be decoded as 0 or 1
        Otherwise, add 'N' to the bitstream (not detected)
        The default is 0.3.

    Returns
    -------
    bitstream : String
        The decoded bitstream

    """
    bitstream = ""
    rmse = feature.rms(S=np.abs(stft(y=audioStream)))
    transientOnsets = onset.onset_detect(audioStream, sampleRate, units="samples", backtrack=True, energy=rmse[0])
    nOnsets = transientOnsets.shape[0]
    # Ignore first and last transient
    for onsetIndex in range(1, nOnsets-1):
        transient = audioStream[transientOnsets[onsetIndex]:transientOnsets[onsetIndex+1]]
        (transient, _) = effects.trim(transient, frame_length=256, hop_length=128)
        
        #plt.figure()
        #display.waveplot(transient, sampleRate)
        
        soundClass = predictSoundClass(transient, model, sampleRate)
        
        if (soundClass < detectionThreshold):
            bitstream += '0'
        elif (soundClass > 1.0 - detectionThreshold):
            bitstream += '1'
        else:
            bitstream += 'N'
    return bitstream