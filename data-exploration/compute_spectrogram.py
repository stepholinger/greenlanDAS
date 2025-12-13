import obspy
from scipy.signal import spectrogram
import numpy as np
import glob
import matplotlib. pyplot as plt
import matplotlib.dates as mdates
import datetime as datetime
import h5py



def compute_spectrogram(file,window_length,noverlap):
    
    # read the file
    st = obspy.read(file)
    fs = st[0].stats.sampling_rate
    
    # filter
    st.filter("bandpass",freqmin=1/window_length,freqmax=0.49*fs)
    
    # make a spectrogram
    f,t,s = spectrogram(st[0].data, fs=fs, nperseg=int(fs*window_length),
                        noverlap=(np.round(noverlap*fs*window_length)))

    # set output parameters
    channel = file.split("_")[1].split(".")[0]
    
    return f,t,s,channel



def compute_and_plot_spectrogram(file,window_length,noverlap,out_path):
    
    # read the file
    st = obspy.read(file)
    fs = st[0].stats.sampling_rate
    
    # filter
    st.filter("bandpass",freqmin=1/window_length,freqmax=0.49*fs)
    
    # make a spectrogram
    f,t,s = spectrogram(st[0].data, fs=fs, nperseg=int(fs*window_length),
                        noverlap=(np.round(noverlap*fs*window_length)))

    # set output parameters
    channel = file.split("_")[1].split(".")[0]
    fname = out_path + "channel_"+channel+".png"
    title = "Channel " + channel
    
    # make a plot and record
    plot_spectrogram(st,f,t,s,title,fname)
    
    return f,t,s,channel
    
    
    
def plot_spectrogram(st,f,t,s,title,fname):
    
    # get metadata
    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime
    hours = np.floor((endtime-starttime)/(6*3600))+1

    # make plot
    fig,ax = plt.subplots(1,1,figsize=[20,7])
 
    # plot spectrogram
    ticks = [starttime.datetime + datetime.timedelta(seconds=hour*6*3600) for hour in range(int(hours))]
    times = [starttime.datetime + datetime.timedelta(seconds=time) for time in t]
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d \n %H:%M'))
    ax.set_xticks(ticks)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlim([times[0]-datetime.timedelta(seconds=10),times[-1]+datetime.timedelta(seconds=10)])
    ax.set_yticks([-3,-2,-1,0,1,2,3])
    ax.set_yticklabels(["$10^{-3}$","$10^{-2}$","$10^{-1}$","$10^{0}$","$10^{1}$","$10^{2}$","$10^{3}$"],fontsize=15)
    ax.set_ylabel("Frequency (Hz)",fontsize=15)
    ax.set_xlabel("Time",fontsize=15)
    plt.subplots_adjust(right=0.85)
    s_fin = s[1:,:][~np.isinf(s[1:,:])]
    max_amp = np.max(np.log10(s_fin))
    min_amp = np.min(np.log10(s_fin[s_fin!=0]))
    amp_range = max_amp - min_amp
    spec = ax.pcolormesh(times, np.log10(f[1:]), np.log10(s[1:,:]),vmin=-1,vmax=4)#,vmin=min_amp+amp_range*0.5,vmax=0.8*max_amp)
    cbar = plt.colorbar(spec)#,ticks=[-10,-8,-6,-4,-2,0])
    #cbar.ax.set_yticklabels(['$10^{-10}$', '$10^{-8}$', '$10^{-6}$', '$10^{-4}$','$10^{-2}$','$10^0$']) 
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("PSD ((1/s)$^2$/Hz)",size=15)
    plt.title(title,fontsize=25)
    plt.savefig(fname)
    plt.close()
    
    
def compute_and_save_spectrogram(file,window_length,noverlap,out_path):
    
    # read the file
    st = obspy.read(file)
    fs = st[0].stats.sampling_rate
    
    # filter
    st.filter("bandpass",freqmin=1/window_length,freqmax=0.49*fs)
    
    # make a spectrogram
    f,t,s = spectrogram(st[0].data, fs=fs, nperseg=int(fs*window_length),
                        noverlap=(np.round(noverlap*fs*window_length)))

    # set output parameters
    channel = file.split("_")[1].split(".")[0]
    
    # save spectrogram data
    save_spectrogram(f,t,s,channel,out_path)
    
    return f,t,s,channel
    
    
def save_spectrogram(f,t,s,channel,out_path):

    fname = out_path + "channel_"+channel+".h5"

    # save array of spectrograms to h5
    hfile = h5py.File(fname, 'w')
    hfile.create_dataset('s', data=s)
    hfile.create_dataset('channel', data=channel)
    hfile.create_dataset('f', data=f)
    hfile.create_dataset('t', data=t)
    hfile.close()