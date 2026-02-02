import obspy_local
import obspy
from obspy_local.obspy_local.io.segy.core import _read_segy
import glob
from scipy.signal import spectrogram, butter, sosfilt, decimate, detrend
from scipy.signal.windows import hann
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import types
import copy
import h5py

def downsample_file(f, out_path, nx_old, nx_new, fs_new):
    try:
        st = _read_segy(f,npts=nx_old)

        # resample
        st = st.resample(fs_new)

        # correct metadata
        st.stats.binary_file_header.number_of_samples_per_data_trace = nx_new
        st.stats.binary_file_header.sample_interval_in_microseconds = int(1/fs_new*1e6)

        # convert to float32
        for tr in st:
            tr.data = tr.data.astype('float32')

        # write file
        st.write(out_path+f.split("/")[-1],format="SEGY")
    except:
        print("Issue processing file " + f)

# list all the 4khz Greenland files
path_4khz = "/1-fnp/petasaur/p-wd03/StoreGlacier/"
files_4khz = glob.glob(path_4khz+"4kHz/*")

# remove any previously-processed files
path_resampled = "/1-fnp/psound/psound-wd3/greenland/resampled/"
out_files = glob.glob(path_resampled+"*")
out_files = [f.split("/")[-1] for f in out_files]

# set parameters
fs_new = 1000
nx_new = 30000
nx_old = 120000

# get list of files to process
arg_list = []
for f in files_4khz:
    if f.split("/")[-1] not in out_files:
        arg_list.append(f)
        
# run in serial
for f in arg_list:
    downsample_file(f, path_resampled, nx_old, nx_new, fs_new)
