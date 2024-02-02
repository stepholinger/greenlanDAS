using SeisNoise, SeisIO, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates, PyCall
include("Types.jl")
include("Nodal.jl")
include("Misc.jl")
include("Workflow.jl")

# list all 1khz and resampled Greenland files
path = "/1-fnp/petasaur/p-wd03/folder/data/"
files = glob("*",path_1khz)

# choose channels
chan_start = 331
chan_end = 2391

# set filter band and resample frequency
freqmin,freqmax = 1,100
fs = 200

# harddcode samples per file (otherwise, this is pulled from the data)
n_per_file = []

# set frequeny and time normalization
whitening = 0
time_norm = "1bit"

# set windowing parameters
cc_len = 10
maxlag = 1

# choose fk filter bounds
cmin,cmax = 750,4250
sgn = "none" # "both", "pos", "neg"

# indicate cable geometry (linear or u-shaped)
geometry = "l"

# choose mode (corr = all cross correlations, auto = autocorrelations only, 
# auto_cross = autocorrelation for each leg of cable with u-shaped geometry and cross-leg pairs)
mode = "corr"

# set substack timing and output path for 1 khz files
substack_time = Minute(60)
Ns,Nf = read_nodal("segy", files[1]), read_nodal("segy", files[end])
start_datetime,end_datetime = get_datetime(Ns),get_datetime(Nf)
output_times = start_datetime+substack_time:substack_time:end_datetime

# set output path
out_path = "/1-fnp/petasaur/p-wd03/folder/data/"

# choose gpu
device = 1

# correlate 1khz files
NC = workflow(files,cc_len,maxlag,freqmin,freqmax,fs,cmin,cmax,sgn,
               time_norm,[chan_start,chan_end],output_times,out_path,geometry,whitening,n_per_file,mode,device)