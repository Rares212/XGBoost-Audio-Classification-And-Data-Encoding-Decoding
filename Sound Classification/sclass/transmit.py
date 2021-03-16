# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:29:14 2021

@author: Rares
"""

import numpy as np

def encodeBitstreamToAudio(bitstream, audio0, audio1, sampleRate=44100, bitDelay=0.5, paddingDelay=2.0, padding=True):
    """
    Encodes a bitstream to an audio stream
    audio0 -> bit 0
    audio1 -> bit 1

    Parameters
    ----------
    bitstream : String
        Input bitstring. 
        Special characters: 'd' -> add padding delay
                            'b' -> add bit delay (smaller)
    audio0 : numpy vector
        Audio array for bit 0
    audio1 : numpy vector
        Audio array for bit 1
    sampleRate : int, optional
        Sample rate in Hz. The default is 44100.
    bitDelay : float, optional
        Delay between bits in seconds. The default is 0.5.
    paddingDelay : float, optional
        Padding delay in seconds. The default is 2.0.
    padding : bool, optional
        Whether to add a padding delay + bit 1 at the beggining and end
        of the audio stream. The default is True.

    Returns
    -------
    audioOutput : np array
        The encoded audio.

    """
    audioOutput = np.empty(0);
    if (padding):
        audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * paddingDelay)))
        audioOutput = np.append(audioOutput, audio1)
        audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * bitDelay)))
    for bit in bitstream:
        if (bit == '0'):
            audioOutput = np.append(audioOutput, audio0)
            audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * bitDelay)))
        elif (bit == '1'):
            audioOutput = np.append(audioOutput, audio1)
            audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * bitDelay)))
        elif (bit == 'd'):
            audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * paddingDelay)))
        elif (bit == 'b'):
            audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * bitDelay)))
    if (padding):
        audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * bitDelay)))
        audioOutput = np.append(audioOutput, audio1)
        audioOutput = np.append(audioOutput, np.zeros(int(sampleRate * paddingDelay)))
    return audioOutput