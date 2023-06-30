using SeisNoise, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData
import SeisIO: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import Base:show, size, summary
include("correlation_codes/Types.jl")
include("correlation_codes/Nodal.jl")
include("correlation_codes/Misc.jl")
include("correlation_codes/Workflow.jl")

# list all 1khz and resampled Greenland files
path_1khz = "/1-fnp/petasaur/p-wd03/greenland/Store Glacier DAS data/"
path_resampled = "/1-fnp/pnwstore1/p-wd05/greenland/resampled/"
files_1khz = glob("1kHz/*",path_1khz)
files_resampled = glob("*",path_resampled)
files = cat(files_1khz,files_resampled,dims=1)
N = read_nodal("segy", files[2])

# choose channels
chan_start = 331
chan_end = 2391
chans = [chan_start,chan_end]

# set filter band
freqmin,freqmax = 1,200
fs = freqmax*2+1

# set time normalization
time_norm = "1bit"

# set windowing parameters
cc_len = 10
maxlag = 1

# choose fk filter bounds
#cmin,cmax = 1000,4000
#sgn = "both"

# set substack timing and output path for 1 khz files
substack_time = Hour(1)
Ns,Nf = read_nodal("segy", files[2]), read_nodal("segy", files[end])
start_datetime,end_datetime = get_datetime(Ns),get_datetime(Nf)
output_times = start_datetime+substack_time:substack_time:end_datetime
out_path = string("/fd1/solinger/correlations/fk_none/")

# correlate 1khz files
# NC = workflow(files[2264:end],cc_len,maxlag,freqmin,freqmax,fs,
#               cmin,cmax,sgn,time_norm,chans,output_times,out_path,30000)
NC = workflow(files[2:end],cc_len,maxlag,freqmin,freqmax,fs,
              time_norm,chans,output_times,out_path,30000)