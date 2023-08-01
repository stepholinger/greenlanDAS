import obspy
import h5py
from scipy.signal import spectrogram, butter, sosfilt, decimate, detrend
from obspy_local.obspy_local.io.segy.core import _read_segy
from scipy.signal.windows import hann
import scipy
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as datetime
import pickle
import collections
import multiprocessing
import types
import copy
import time


def plot_moveout(data,st,channels):
    t = st[0].times()
    spacing = 1.017
    datetimes = [st[0].stats.starttime.datetime+datetime.timedelta(seconds=sec) for sec in t]
    fig, ax = plt.subplots(figsize=(15,10))
    ax.pcolormesh(datetimes,(np.array(channels)-channels[0])*spacing,np.transpose(data),cmap='gray',vmin=-50,vmax=50)
    plt.gca().invert_yaxis()
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Time (s)")
    return ax


# i am not positive this works
def read_segy_direct(fname,channels,ns,fs,ns_read=-1,offset=0):
    data = np.zeros((ns,len(channels)))
    with open(fname,'br') as f:
        for i,channel in enumerate(channels):
            f.seek(3600 + 240*(channel+1) + 4*ns*channel + 4*offset*fs,0)
            if ns_read == -1:
                ns_read = ns
            data[:,i] = struct.unpack('%df' % ns_read,f.read(4*ns_read))
    return data


def read_data(file,channels):
    st = _read_segy(file)
    t = st[0].times(type="utcdatetime")
    data = np.zeros((st[0].stats.npts,len(channels)))
    for i,channel in enumerate(channels):
        data[:,i] = st[channel].data
    return data,t


def design_butter(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    
def filter_butter(data, lowcut, highcut, fs, order=4):
        sos = design_butter(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data, axis=0)
        return y
    
    
def taper(data,taper_length,fs):
    taper_length = int(taper_length*2*fs)
    window_ends = hann(taper_length*2)
    window = np.ones(data.shape)
    for i in range(window.shape[1]):
        window[0:taper_length,i] = window_ends[0:taper_length]
        window[-taper_length:window.shape[0],i] = window_ends[-taper_length:window.shape[0]]
    data = window * data
    return data


def get_data(st,s):
    data = np.zeros((s.file_length*s.fs,len(s.channels)))
    for i,channel in enumerate(s.channels):
        data[:,i] = st.select(station=str(channel))[0].data
    return data


def save_stream_h5(data,st,s):
    
    # get start and endtime for saving
    startstr = st[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")
    endstr = st[0].stats.endtime.strftime("%Y-%m-%dT%H:%M:%S")
    
    # save result for this file
    fname = (s.out_path+"chans"+str(s.channels[0])+"-"+str(s.channels[-1])+
            "_fs"+str(int(s.fs))+"_"+startstr+"-"+endstr+".h5")
    
     # save array of spectrograms to h5
    with h5py.File(fname,"w") as f:
        f["data"] = data
        f.attrs["starttime"] = st[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S%f")
        f.attrs["endtime"] = st[0].stats.endtime.strftime("%Y-%m-%dT%H:%M:%S%f")
    f.close()
    

def save_data_h5(s):
    
    # get start and endtime for saving
    startstr = s.t[0].datetime.strftime("%Y-%m-%dT%H:%M:%S")
    endstr = s.t[-1].datetime.strftime("%Y-%m-%dT%H:%M:%S")
    timestamps = [time.datetime.timestamp() for time in s.t]
    
    # save result for this file
    fname = (s.out_path+"chans"+str(s.channels[0])+"-"+str(s.channels[-1])+
            "_fs"+str(int(s.fs_new))+"_"+startstr+"-"+endstr+".h5")
    
    # save array of data to h5
    with h5py.File(fname,"w") as f:
        f["data"] = s.data
        f["timestamps"] = timestamps
        f.attrs["starttime"] = s.t[0].datetime.strftime("%Y-%m-%dT%H:%M:%S%f")
        f.attrs["endtime"] = s.t[-1].datetime.strftime("%Y-%m-%dT%H:%M:%S%f")
    f.close()
    
    
def read_and_preprocess(s):
    try:
        # read the file and preprocess
        data,s.t = read_data(s.file,s.channels)
        data = preprocess_data(data,s)
        s.data = data
    except:
        print("Issue processing file " + f)
        s.data = []
        s.t = []
        
    return s


def preprocess_data(data,s):
    data = detrend(data, axis=0, type='linear')
    data = taper(data,s.taper_length,s.fs)
    sos = design_butter(s.freq[0],s.freq[1],s.fs)
    data = filter_butter(data,s.freq[0],s.freq[1],s.fs)
    s.fs_new = s.freq[1]*2
    data = decimate(data,int(s.fs/s.fs_new),axis=0)
    return data
        
    
def save_dataset_h5(s):
    
    # read the relevant files, fill metadata, taper, and filter
    inputs = []
    for f in s.files:
        
        # add to preprocessing input list
        s.file = f
        inputs.append(copy.deepcopy(s))

    # map inputs to read_preprocess_save as each call finishes
    multiprocessing.freeze_support()
    p = multiprocessing.Pool(processes=s.n_procs)
    for result in p.imap_unordered(read_and_preprocess,inputs):
        if len(result.data) != 0:
            save_data_h5(result)
    p.close()
    p.join()               