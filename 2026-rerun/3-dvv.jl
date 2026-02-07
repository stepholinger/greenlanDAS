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
include("../correlation/functions/Types.jl")
include("../correlation/functions/Nodal.jl")
include("../correlation/functions/Misc.jl")
include("../correlation/functions/Workflow.jl")
include("../correlation/functions/Dvv.jl")

#------CONSOLIDATE AUTOCORRELATIONS-------

# set path to correlation files
path = string("/1-fnp/psound/psound-wd3/greenland/correlations/fk_1500_2250/no_whitening/10_min/icequakes_removed/")

# set number of samples and correlations
ns = 801
ncorr = 3090

# restack time
restack_interval = 10

# which fk band?
clims = [[3500,4250],[1500,2250]]

# frequency bands to return and which bands to process later (one at a time; use 1 if only ran for a single band)
bands = [[10. 13.];]
band = 1

# consolidate autocorrelations into one file
for clim in clims:
  
  # get files and datetimes
  files, path, datetimes = get_files_and_datetimes(clim[1],clim[2])

  # get autocorrs
  restack_times, all_autocorrs = collect_autocorrs(files,datetimes,restack_interval,ns,ncorr,bands)
  
  # save the autocorrelations and datetime information
  fname = string(path,"autocorrelations_",Int(bands[1,1]),"-",Int(bands[end,2]),"Hz_",restack_interval,"_min_stack.jld2")
  JLD2.save(fname,Dict("autocorrs"=>all_autocorrs))
  fname = string(path,"datetimes_",restack_interval,"_min_stack.jld2")
  JLD2.save(fname,Dict("datetimes"=>collect(restack_times)))

#------NOW LETS SET UP THE DVV PARAMETERS-------

# set parameters for dv/v
gain_correct = "none"
method = "stretching"
dvlims = [-0.2,0.2]
ntrial = 100
ref_type = ["time_avg"]
tlims = [[0.3,0.8],[0.0,0.3]]
restack_intervals = [10]
sides = ["pos","neg"]
modes = ["dx"]

# choose any time periods to skip due to gaps, icequakes, etc
skip_time = [DateTime(2019,7,7,13),DateTime(2019,7,7,14)]

# set up parameter space (useful if you are testing a bunch of options e.g. dv/v method, reference, etc).
parameters = collect(Iterators.product(clims,sides,tlims,restack_intervals,ref_type,modes))
parameters = filter(e->e[1] == clims[1] && e[3] == tlims[1] || 
                    e[1] == clims[2] && e[3] == tlims[2],parameters)
parameters = filter(e->e[6] == modes[1] && e[5] == ref_type[1] || 
                    e[6] == modes[2] && e[5] == ref_type[2],parameters)

#------ACTUALLY RUN THE DVV------

# iterate through list of parameters
for parameter in parameters
    
    # get parameters
    clim = parameter[1]
    side = parameter[2]
    tlim = parameter[3]
    restack_interval = parameter[4]
    ref_type = parameter[5]
    mode = parameter[6]

    # choose first hour to consider (exclude noisy period at start of deployment)
    start_time = DateTime(2019,7,6,10)

    # load autocorrelations
    path = string("/fd1/solinger/correlations/fk_",clim[1],"_",clim[2],"/no_whitening/10_min/icequakes_removed/")
    fname = string(path,"autocorrelations_",Int(bands[1,1]),"-",Int(bands[end,2]),"Hz_",restack_interval,"_min_stack.jld2")
    autocorrs = JLD2.load(fname)["autocorrs"]

    # get correct side
    zero_ind = size(autocorrs,1)÷2+1
    if side == "pos"
        autocorrs = autocorrs[zero_ind:end,:,:,:]
    elseif side == "neg"
        autocorrs = autocorrs[1:zero_ind,:,:,:]
        autocorrs = reverse(autocorrs,dims=1)
    end

    # read datetimes
    fname = string(path,"datetimes_",restack_interval,"_min_stack.jld2")
    datetimes = JLD2.load(fname)["datetimes"]

    # zero the window to skip
    skip_ind = [findfirst(datetimes .> skip_time[1]),findfirst(datetimes .> skip_time[2])]
    autocorrs[:,:,skip_ind[1]:skip_ind[2]] .= 0

    # get index of first correlation
    start_ind = findfirst(datetimes .> start_time)
    autocorrs = autocorrs[:,:,start_ind:end,:]

    # compute reference waveforms based on selected parameters
    if mode == "dt"
        if ref_type == "time_avg"
            ref_corrs = sum(autocorrs[:,:,:,:],dims=3)
        elseif ref_type == "last"
            last_idx = findall(x -> x == 1, (autocorrs[1,1,:,1] .!= 0))[end]
            ref_corrs = autocorrs[:,:,last_idx:last_idx,:]
        end
    elseif mode == "dx"
        if ref_type == "time_space_avg"
            time_avg_corrs = mean(autocorrs[:,:,(autocorrs[1,1,:,1] .!= 0),1],dims=3)
            time_space_avg_corrs = mean(time_avg_corrs,dims=2)
            ref_corrs = autocorrs[:,:,1:1,:]
            ref_corrs .= time_space_avg_corrs
        end
    end

    # run stretching dvv computation
    if mode == "dt"
        dvv = compute_stretching_dvdt(autocorrs,ref_corrs,bands,tlim,dvlims,ntrial,gain_correct)
    elseif mode == "dx"
        if ref_type == "time_space_avg"
            time_avg_autocorrs = mean(autocorrs[:,:,(autocorrs[1,1,:,1] .!= 0),1],dims=3)
            dvv = compute_stretching_dvdx(time_avg_autocorrs,ref_corrs,bands,tlim,dvlims,ntrial,gain_correct)
        elseif ref_type == "cumulative"
            dvv = compute_stretching_dvdx_cumulative(autocorrs,bands,tlim,dvlims,ntrial,gain_correct)
        end
    end

    # save results
    fname = string(path,"auto_dv",mode,"_",method,"_",Int(bands[1,1]),"-",Int(bands[end,2]),"Hz_",tlim[1],"-",tlim[2],"s_",restack_interval,"_min_",ref_type,"_ref_",side,".jld2")
    JLD2.save(fname,Dict("dvv"=>dvv))

    # give output
    println("Saved ",fname)
end
