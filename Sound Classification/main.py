# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:07:46 2021

@author: Rares
"""

import sclass
import librosa
import matplotlib.pyplot as plt
from librosa import util, display, effects
import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
 
    # File paths
    dirName = os.path.dirname(__file__)
    kicksFolder = os.path.join(dirName, "Dataset/Kicks")
    snaresFolder = os.path.join(dirName, "Dataset/Snares")
    kickFile = os.path.join(dirName, "Kick Subby 1.wav")
    snareFile =os.path.join(dirName, "Snare Test.wav")
    modelFile = os.path.join(dirName, "Models\\Binary Model")
     
    # Audio data
    nCoeffs = 32
    sampleRate = 44100
    
    # Load dataset as features
    kickFeatures = sclass.loadAndGetMfcc(kicksFolder, nCoeffs=nCoeffs)
    snareFeatures = sclass.loadAndGetMfcc(snaresFolder, nCoeffs=nCoeffs)
    (X, y) = sclass.combineFeatures(kickFeatures, snareFeatures)
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2, stratify=y)
    
    # Train the XGBoost model | comment if already trained
    # sclass.trainModel(trainX, testX, trainY, testY, modelFile)
    
    # Load the XGBoost model
    boostModel = xgb.Booster()
    boostModel.load_model(modelFile + ".model")
    
    # Plot feature importance
    featureImportance = boostModel.get_score()
    featureNames = list(featureImportance.keys())
    featureValues = list(featureImportance.values())
    featureData = pd.DataFrame(data=featureValues, index=featureNames,
                               columns=["score"]).sort_values(by = "score", ascending=False)
    featureData.plot(kind='barh')

    # Plot confusion matrix
    predictedY = boostModel.predict(xgb.DMatrix(testX))
    confMatrix = confusion_matrix(testY, np.rint(predictedY), labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=["Kick", "Snare"])
    disp.plot()
    
    # Load, normalize and trim the kick/snare test files
    (kickSound, _) = librosa.load(kickFile, sampleRate)
    (snareSound, _) = librosa.load(snareFile, sampleRate)
    kickSound = util.normalize(kickSound)
    snareSound = util.normalize(snareSound)
    (kickSound, _) = effects.trim(kickSound)
    (snareSound, _) = effects.trim(snareSound)
    
    # Encode the bitstream - 0 -> kick, 1 -> snare, d -> delay
    # Add a transient at the beggining and the end as
    # stream START and STOP flags
    testStream = sclass.encodeBitstreamToAudio("01011100", kickSound, snareSound, sampleRate, 
                                               bitDelay=0.1, paddingDelay=0.2, padding=True)
    plt.figure()
    display.waveplot(testStream, sampleRate)
    
    # Play the bitstream audio
    sd.default.samplerate = sampleRate
    sd.play(testStream)
    sd.wait()
    
    # Decode the bitstream
    decodedBitstream = sclass.decodeBitstreamFromAudio(testStream, boostModel, sampleRate)
    print("Decoded bitstream: " + decodedBitstream)
    
    