"""
Name: Jose Varela
Email: jvarela@haverford.edu

This file does the feature extraction for MFCCs, chromagrams, and spectral features.
*Note* it accounts for MFCCs, chromagrams, and spectral features all in one function. 
"""
import numpy as np
import librosa
from config import Sample_Rate, Duration, N_MFCC

def feat_extract(y, sr=Sample_Rate): 
    """This function extracts MFCC, chromagram, and spectral features"""

    #Ensure Mono
    if y.ndim > 1:
        y = librosa.to_mono(y)

    #Trimming/padding to exactly 30 seconds
    expected_len = Sample_Rate * Duration
    if len(y) > expected_len:
        y = y[:expected_len]
    elif len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    #MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)

    #Chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    #For the spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    # Combining everything into one vector
    feature_vector = np.hstack([
        mfcc_mean,
        chroma_mean,
        spec_centroid,
        spec_bandwidth,
        spec_rolloff,
        zcr,
    ])
    #Return the feature vector with all of the features combines
    return feature_vector
