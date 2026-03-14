using SeisNoise, SeisBase, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData 
import SeisBase: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import DSP: hilbert
import Images: findlocalmaxima
import Base:show, size, summary
import PyCall
import SeisDvv
import Dates
import FiniteDifferences
import CSV
using NetCDF
using DataFrames
include("functions/Types.jl")
include("functions/Nodal.jl")
include("functions/Misc.jl")
include("functions/Workflow.jl")

# processing parameters
freqmin, freqmax = 20,50
sgn = "both"
cmin,cmax = 750,4250
maxlag = 1

# choose channels
chan_start = 331
chan_end = 1361
chans = [chan_start,chan_end]

# list correlation files
path = string("/1-fnp/psound/psound-wd3/greenland/correlations/fk_",cmin,"_",cmax,"/no_whitening/10_min/icequakes_removed/750m_cross_correlations/")
files = glob("*",path)
files = files[BitVector(1 .- contains.(files,"error"))]
files = files[BitVector(1 .- contains.(files,"autocorrelation"))]
files = files[BitVector(1 .- contains.(files,"datetimes"))]
files = files[BitVector(1 .- contains.(files,"dv"))]
files = files[BitVector(1 .- contains.(files,"stack"))]

# subset further in time
files = files[20:end] 

# read results and stack
C = load(files[1])["NodalCorrData"]
for f in files
    try
    	println("Stacked ",f)
    	C.corr = C.corr + JLD2.load(f)["NodalCorrData"].corr
    catch error
	bt = backtrace()
        msg = sprint(showerror, error, bt)
	error_string = "\nError on file: "*f*"\n"*msg*"\n"
	print(error_string)
    end
end

# save stack
fname = string(path,"stack.jld2")
JLD2.save(fname,Dict("NodalCorrData"=>C))

# postprocessing
C_filt = deepcopy(C)
clean_up!(C_filt,freqmin,freqmax)

# normalize post-stack
abs_max!(C_filt)

fname = string(path,freqmin,"_",freqmax,"Hz_stack_normalized.jld2")
JLD2.save(fname,Dict("NodalCorrData"=>C_filt))

