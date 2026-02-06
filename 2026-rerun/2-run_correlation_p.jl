using SeisNoise, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates, PyCall
import SeisNoise: NoiseData
import SeisBase: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import Base:show, size, summary
include("functions/Types.jl")
include("functions/Nodal.jl")
include("functions/Misc.jl")
include("functions/Workflow.jl")

# list all 1khz and resampled Greenland files
path_1khz = "/1-fnp/petasaur/p-wd03/StoreGlacier/"
path_resampled = "/1-fnp/psound/psound-wd3/greenland/resampled"
files_1khz = glob("1kHz/*",path_1khz)
files_resampled = glob("*",path_resampled)
files = cat(files_1khz,files_resampled,dims=1)
N = read_nodal("segy", files[1])

# choose channels
chan_start = 331
chan_end = 2391
chans = [chan_start,chan_end]

# set filter band
freqmin,freqmax = 1,100
fs = 400

# set frequeny and time normalization
whitening = 0
time_norm = "1bit"

# set windowing parameters
cc_len = 10
maxlag = 1

# choose fk filter bounds
cmin,cmax = 3500,4250
sgn = "pos"

# indicate cable geometry (linear or u-shaped)
geometry = "u"

# subset to some specific files
files = files[2:end]

# define function for reading pickle using PyCall
py"""
import pickle
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
load_pickle = py"load_pickle"

# load detection times and predictions
detection_file = "detection_times.pickle"
prediction_file = "detection_predictions.pickle"
detection_times = load_pickle(detection_file)
predictions = load_pickle(prediction_file)

# consolidate detection times and predictions into two lists
all_detections = []
for det in detection_times
    for d in det
        global all_detections = vcat(all_detections,d)
    end
end
all_predictions = []
for pred in predictions
    for p in pred
        global all_predictions = vcat(all_predictions,p)
    end
end

# get all detection times with a prediction over 300
detection_times_skip = all_detections[all_predictions .> 300]

# get datetimes of each file
file_datetimes = []
for i=1:size(files,1)
    datetime = DateTime(string("20"*split(split(files[i],"AQ_")[2],".sgy")[1]),dateformat"yyyymmddHHMMSS")
    global file_datetimes = vcat(file_datetimes,datetime)
end

# get list of files containing detections (files to skip)
files_to_skip = []
for i = 1:size(detection_times_skip,1)
    time_diffs = detection_times_skip[i] .- file_datetimes
    time_diffs[time_diffs .< Millisecond(0)] .= Millisecond(1e10)
    file_to_skip = files[time_diffs .== minimum(time_diffs)]
    global files_to_skip = vcat(files_to_skip,file_to_skip)
end

# remove those files from list 
files = setdiff(files,files_to_skip)
println("Running on ",string(size(files,1))," files")

# set substack timing and output path for 1 khz files
substack_time = Minute(10)
Ns,Nf = read_nodal("segy", files[1]), read_nodal("segy", files[end])
start_datetime,end_datetime = get_datetime(Ns),get_datetime(Nf)
output_times = start_datetime+substack_time:substack_time:end_datetime
out_path = string("/1-fnp/psound/psound-wd3/greenland/correlations/fk_3500_4250/no_whitening/10_min/icequakes_removed/")

# correlate 1khz files
NC = workflow(files[2984:end],cc_len,maxlag,freqmin,freqmax,fs,cmin,cmax,sgn,
               time_norm,chans,output_times,out_path,geometry,whitening,30000,"auto_cross",0)
