#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
When using this resource, please cite the original publication:
Aysan Mahmoudzadeh, Iman Azimi, Amir M. Rahmani, and Pasi Liljeberg, “Lightweight Photoplethysmography Quality Assessment for Real-time IoT-based Health Monitoring using Unsupervised Anomaly Detection,” Elsevier International Conference on Ambient Systems, Networks and Technologies (ANT’21), 2021, Poland.


authors: Aysan Mahmoudzade and Iman Azimi

"""




import pandas as pd
import numpy as np
import os
import sys
from scipy import signal, stats
from sklearn.covariance import EllipticEnvelope

def butter_filtering(sig,fs,fc,order,btype): 
    #Parameters: signal, sampling frequency, cutoff frequencies, order of the filter, filter type (e.g., bandpass)
    #Returns: filtered signal
    w = np.array(fc)/(fs/2)
    b, a = signal.butter(order, w, btype =btype, analog=False)
    filtered = signal.filtfilt(b, a, sig)
    return(filtered)

def peak_detection(sig,fs):
    #Parameters: signal, sampling frequency
    #Returns: peaks indices
    ## peaks in raw signal
    NN_index_sig = np.array(signal.argrelextrema(sig, np.greater)).reshape(1,-1)[0]
    ## filter the signal based on the frequancy of HR
    f, ppg_den = signal.periodogram(sig, fs)
    min_f = np.where(f >= 0.6)[0][0] #minimum heart rate frequency
    max_f = np.where(f >= 3.0)[0][0] #maximum heart rate frequency
    ppg_HR_freq = ppg_den[min_f:max_f]
    HR_freq = f[min_f:max_f]
    # Define cut-off frequancy
    HRf = HR_freq[np.argmax(ppg_HR_freq)]
    boundary = 0.5
    if HRf - boundary > 0.6:
        HRfmin = HRf - boundary
    else:
        HRfmin = 0.6
    if HRf + boundary < 3.0:
        HRfmax = HRf + boundary
    else:
        HRfmax = 3.0
    filtered = butter_filtering(sig,fs,np.array([HRfmin,HRfmax]),5,'bandpass')
    ## peaks in filtered signal
    NN_index_filtered = np.array(signal.argrelextrema(filtered, np.greater)).reshape(1,-1)[0]
    ## find the correct peaks according to peaks of raw and filtered
    NN_index = np.array([]).astype(int)
    for i in NN_index_filtered:
        NN_index = np.append(NN_index,NN_index_sig[np.abs(i - NN_index_sig).argmin()])
    NN_index = np.unique(NN_index)
    return(NN_index) 

def troughs_detection(sig,NN_index):
    #Inputs: signal, peaks indices
    #Returns: troughs indices
    MM_index = np.array([]).astype(int)
    for i in range(NN_index.shape[0]-1):
        MM_index = np.append(MM_index,np.argmin(sig[NN_index[i]:NN_index[i+1]]) + NN_index[i])
    return(MM_index)

def approx_entropy(U, m, r) -> float:
    #Inputs: signal, m, r
    #Returnts: approximate entropy
    U = np.array(U)
    N = U.shape[0]
            
    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i+m] for i in range(int(z))])
        if z==0:
            return 10
        else:
            X = np.repeat(x[:, np.newaxis], 1, axis=2)
            C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
            return np.log(C).sum() / z
    
    return abs(_phi(m + 1) - _phi(m))

def shan_entropy(sig):
    #Inputs: signal
    #Returnts: shannon entropy
    hist = np.histogram(sig,10)[0]
    pk = hist/hist.sum()
    return stats.entropy(pk,base=2)

def spec_entropy(sig,fs):   
    #Inputs: signal, sampling frequency
    #Returnts: spectral entropy
    _, Pxx_den = signal.welch(sig, fs)
    pk = Pxx_den/np.sum(Pxx_den)
    return stats.entropy(pk, base=2)
    
    
def feature_extraction(sig,fs):
    #Inputs: signal, sampling frequency
    #Returns: features: i.e., skewness, kurtosis, approximate entropy, shannon entropy, spectral entropy
    peaks_idx = peak_detection(sig,fs)
    troughs_idx = troughs_detection(sig,peaks_idx)
    f_skewness = []
    f_kurtosis = []
    f_approx_entropy = []
    for i in range(troughs_idx.size-1):
        heart_cycle =  sig[troughs_idx[i]:troughs_idx[i+1]]
        f_skewness.append(stats.skew(heart_cycle))
        f_kurtosis.append(stats.kurtosis(heart_cycle))
        f_approx_entropy.append(approx_entropy(heart_cycle, 2, 0.1*np.std(heart_cycle)))
    f_skewness = max(f_skewness)-min(f_skewness)
    f_kurtosis = max(f_kurtosis)-min(f_kurtosis)
    f_approx_entropy = max(f_approx_entropy)-min(f_approx_entropy)
    f_shan_entropy = shan_entropy(sig)
    f_spec_entropy = spec_entropy(sig,fs)
    return([f_skewness,f_kurtosis,f_approx_entropy,f_shan_entropy,f_spec_entropy])

if __name__ == '__main__':
    fs = 20.0 #Hz
    load_dir = "data/" #PPG samples
    filenames = os.listdir(load_dir)
    if not filenames:
        sys.exit('No file in the directory')
    feature_list = []
    for f in filenames:
        df = pd.read_csv(load_dir + f)
        ts = df['timestamp'].values
        ppg = df['ppg'].values
        ##feature extraction
        ppg_filtered = butter_filtering(ppg,fs,[0.6,3.0],5,'bandpass')
        feature_list.append(feature_extraction(ppg_filtered,fs))
    
    ##model - training phase
    X_train_mean = np.mean(np.array(feature_list), axis=0)
    X_train_std = np.std(np.array(feature_list), axis=0)
    X_train = (np.array(feature_list) - X_train_mean)/X_train_std    
    model_EE = EllipticEnvelope(contamination=0.34)
    model_EE.fit(np.array(feature_list))
    
    ##model - test phase
    #X_test = (np.array(feature_list) - X_train_mean)/X_train_std
    #yhat = model_EE.predict(X_test)
    
        
        
        
        
        
    
