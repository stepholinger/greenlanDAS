import h5py
import numpy as np
import glob
import datetime as datetime
import pickle
import collections
import multiprocessing
import types
import copy
import time


def correlate(s1,s2,mode="same"):

    # throw an error of input sizes are inconsistent
    if s1.shape != s2.shape:
        raise ValueError("s1 and s2 must have the same size!")
    
    # get fft size
    sz = s1.shape[0]
    n_bits = 1+int(np.log2(2*sz-1))
    fft_sz = 2**n_bits
    
    # take FFT along time axis for both
    fft_s1 = np.fft.fft(s1, fft_sz, axis=0)
    fft_s2 = np.fft.fft(s2, fft_sz, axis=0)
        
    # take complex conjugate of second signal 
    fft_s2_conj = np.conj(fft_s2)
      
    # multiply to get correlation function
    corr_fft = fft_s1*fft_s2_conj
    
    # take inverse fourier transform
    corr = np.fft.ifft(corr_fft, axis=0)
    
    # normalize using the magnitude of both input data
    norm1 = np.linalg.norm(s1,axis=0)
    norm2 = np.linalg.norm(s2,axis=0)
    norm_factor = norm1*norm2
    corr = np.vstack((corr[-(sz-1) :], corr[:sz]))
    norm_corr = np.real(corr) / norm_factor
    
    # return desired part of correlation function
    if mode == "full":
        pass
    elif mode == "same":
        norm_corr = norm_corr[int(sz/2):-int(sz/2)+1]

    return norm_corr


def correlate_np(template,data):
    
    # define container
    corrs = []
    
    # iterate through each channel
    for i in range(template.shape[1]):
        corr = np.correlate(template[:,i],data[:,i],"same")
        norm1 = np.linalg.norm(template[:,i],axis=0)
        norm2 = np.linalg.norm(data[:,i],axis=0)
        norm_factor = norm1*norm2
        corrs.append(corr/norm_factor)
    corrs = np.stack(corrs)
    return corrs


def window_and_correlate(template,data):

    # define container
    all_corr = []

    # get some helpful values
    window_length = template.shape[0]
    num_windows = int(data.shape[0]/window_length)

    # iterate through time windows
    for i in range(num_windows):

        # pull out a time window of data
        start_index = i*window_length
        end_index = start_index + window_length
        window = data[start_index:end_index,:]
        
        # call cross correlation function
        corr = correlate_fft(template,window)

        # save value
        all_corr.append(corr)
    
    # reshape output
    all_corr = np.stack(all_corr)

    return all_corr


def correlate_templates(c):

    # read the relevant preprocessed file
    with h5py.File(c.file,"r") as f:
        data = f['data'][()]

        # run template matching code for each template
        c.corrs = []
        for template in c.templates:
            c.corrs.append(window_and_correlate(template,data))

        # get start and endtime for saving
        c.datestring = c.file.split("_")[-1].split(".h5")[0]
    f.close()
    return c


def template_match(c):

    # construct iterable list of detection parameter objects for imap
    inputs = []
    for f in c.files:

        # add to detection input list
        c.file = f
        inputs.append(copy.deepcopy(c))

#         result = correlate_templates(c)
#         save_correlations(result)
    
    # map inputs to correlate_templates and save as each call finishes
    # THIS MAY NOT WORK- CAN REMOVE PROTECTION BUT THEN WILL HANG ON ERRORS
    if __name__ == '__main__':

        multiprocessing.freeze_support()
        p = multiprocessing.Pool(processes=c.n_procs)
        for result in p.imap_unordered(correlate_templates,inputs):
            save_correlations(result)
        p.close()
        p.join()


def save_correlations(c):
    
    # save result for this file
    fname = (c.out_path + c.datestring + "_corr.h5")
 
    # save array of correlations to h5
    with h5py.File(fname,"w") as f:
        for i,corr in enumerate(c.corrs):
            name = "template_"+str(i)+"_corr"
            f[name] = corr
    f.close()