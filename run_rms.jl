using SeisNoise, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData
import SeisIO: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, 
       show_str, show_t, show_x, show_os, timestamp
import FFTW: rfft, irfft
import Base:show, size, summary
include("correlation_codes/Types.jl")
include("correlation_codes/Nodal.jl")
include("correlation_codes/Misc.jl")

# list all 1khz and resampled Greenland files
path_1khz = "/1-fnp/petasaur/p-wd03/greenland/Store Glacier DAS data/"
path_resampled = "/1-fnp/pnwstore1/p-wd05/greenland/resampled/"
files_1khz = glob("1kHz/*",path_1khz)
files_resampled = glob("*",path_resampled)
files = cat(files_1khz,files_resampled,dims=1)
N = read_nodal("segy", files[2])

# choose channels
chan_start = 333
chan_end = Int64(N.n-chan_start+1)
chans = [chan_start,chan_end]

# set filter band
freq = 0.25
fs = freq*2+1

# set out path
out_path = "/fd1/solinger/rms/below_0.25Hz/"

# carry out running rms computation
rms = compute_rms(files[2:end],freq,"lowpass",fs,chans,out_path,30000,100)