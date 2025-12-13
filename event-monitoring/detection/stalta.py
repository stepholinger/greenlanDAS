import obspy
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, coincidence_trigger
from obspy_local.obspy_local.io.segy.core import _read_segy
import scipy
import numpy as np
import glob
import datetime as datetime
import pickle
import collections
import multiprocessing
import types
import copy


def preprocess(st,channels,freq):
    st = prune_channels(st,channels)
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=1.)
    st.filter("bandpass",freqmin=freq[0],freqmax=freq[1])
    fs_new = int(round(freq[1]*2.1))
    st.resample(210.)
    for tr in st:
        tr.stats.sampling_rate = fs_new
        tr.stats.delta = 1/fs_new
    return st
    
    
def prune_channels(st,channels):
    st = st[channels[0]-1:channels[-1]]
    for i in range(len(st)):
        st[i].stats.station = str(channels[i])
    return st


def stalta(st,sta,lta,channels):   

    # get helpful metadata
    fs = st[0].stats.sampling_rate

    # make containers to store stalta results
    st_stalta = obspy.Stream()
    
    # run stalta on each channel
    for channel in channels:
        tr = st.select(station=str(channel))[0]
        stalta = recursive_sta_lta(tr, int(sta*fs), int(lta*fs))
        
        # fill first 1.5*nstla samples with ones to avoid window-start detections
        stalta[:int(1.5*lta*fs)] = 1
        
        # fill arrays
        tr_stalta = obspy.Trace(data=stalta)
        tr_stalta.stats = tr.stats
        st_stalta += tr_stalta
        
    return st_stalta


def stalta_gap(st,sta,lta,channels,gaps):

    # get gap info
    gap_start = gaps[0][4]
    gap_end = gaps[0][5]

    # trim into stream before and after gap
    st_before_gap = st.copy().trim(endtime=gap_start)
    st_after_gap = st.copy().trim(starttime=gap_end)

    # run stalta algorithm on each channel, before and after gap
    st_stalta = stalta(st_before_gap,sta,lta,channels)
    st_stalta += stalta(st_after_gap,sta,lta,channels)
    st_stalta.merge()
    
    return st_stalta


def detect(d):

    # read the relevant files, fill metadata, taper, and filter
    st = obspy.Stream()
    for f in d.batch_files:
        try:
            # read the file and fill with channel/station metadata
            st_file = _read_segy(f)
            st_file = preprocess(st_file,d.channels,d.freq)
            st += st_file
        except:
            print("Issue processing file " + f)
            continue

    # merge in time
    st.merge()

    # check for gaps- this will probably fail if there is more than one gap
    gaps = st[0:1].get_gaps()
    if gaps:
        st_stalta = stalta_gap(st,d.sta,d.lta,d.channels,gaps)
    else:
        st_stalta = stalta(st,d.sta,d.lta,d.channels)

    # get events using coincidence trigger
    d.triggers = coincidence_trigger(None,d.thr_on,d.thr_off,st_stalta,d.thr_coincidence_sum,
                                     trigger_off_extension=d.trigger_off_extension)

    # get start and endtime for saving
    d.starttime = st[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")
    d.endtime = st[0].stats.endtime.strftime("%Y-%m-%dT%H:%M:%S")

    return d


def save_detections(d):
    
    # save result for this file
    fname = (d.out_path + d.starttime + "-" + d.endtime + "_detections.pickle")
    with open(fname, "wb") as handle:
        pickle.dump(d.triggers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def stalta_detector(d):

    # calculate some batching parameters
    files_per_batch = int(d.batch_length/d.file_length)
    num_batches = int(len(d.files)//files_per_batch)
    final_batch_size = len(d.files)%files_per_batch
    if final_batch_size > 0:
        num_batches += 1
        
    # construct iterable list of detection parameter objects for imap
    inputs = []
    for b in range(num_batches):

        # get start file for this batch
        start_file = b*files_per_batch

        # handle last batch
        if b == num_batches:
            batch_files = d.files[start_file:start_file+final_batch_size]
        else:
            batch_files = d.files[start_file:start_file+files_per_batch]

        # add to detection input list
        d.batch_files = batch_files
        inputs.append(copy.deepcopy(d))

        #result = detect(d)
        #save_detections(result)
    
    # map inputs to stalta_detector and save as each call finishes
    
    if __name__ == '__main__':

        multiprocessing.freeze_support()
        p = multiprocessing.Pool(processes=d.n_procs)
        for result in p.imap_unordered(detect,inputs):
            save_detections(result)
        p.close()
        p.join()