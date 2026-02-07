using SeisNoise, SeisBase, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData 
import SeisBase: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import DSP: hilbert
import Images: findlocalmaxima
import Base:show, size, summary
import SeisDvv
import Dates
import FiniteDifferences
import CSV
using NetCDF
using DataFrames
include("Types.jl")
include("Nodal.jl")
include("Misc.jl")
include("Workflow.jl")

function collect_autocorrs(files,datetimes,restack_interval,ns,ncorr,bands)

    # get start and end datetime
    start_datetime = datetimes[1]-Minute(datetimes[1])-Second(datetimes[1])
    end_datetime = datetimes[end]+Hour(1)-Minute(datetimes[end])-Second(datetimes[end])
    
    # get times to restack
    restack_interval = Minute(restack_interval)
    restack_lims = start_datetime:restack_interval:end_datetime
    restack_times = collect(restack_lims[2:end])

    # get a list of files that should be stacked together
    stack_files = []
    for i=1:size(restack_lims,1)-1
        stack_ind = (datetimes .> restack_lims[i]) .& (datetimes .< restack_lims[i+1])
        append!(stack_files,[files[stack_ind]])
    end
    nstacks = size(stack_files,1)

    # make output data matrix
    all_autocorrs = zeros(ns,ncorr,nstacks,size(bands,1))

    # set a useful counter and read first file
    C = JLD2.load(files[1])["NodalCorrData"]
    for i=1:size(stack_files,1)

        # if there's not a gap
        if size(stack_files[i],1) > 0

            # stack each file
            for j=1:size(stack_files[i],1)

                try
                    if j == 1
                        C = JLD2.load(stack_files[i][j])["NodalCorrData"]
                    else
                        C.corr += JLD2.load(stack_files[i][j])["NodalCorrData"].corr
                    end
                catch error
                    bt = backtrace()
                    msg = sprint(showerror, error, bt)
                    error_string = "\nError on file: "*files[i]*"\n"*msg*"\n"
                    println(error_string)
                end
            end

            # load stack into data structure
            for k=1:size(bands,1)

                # postprocessing
                C_filt = deepcopy(C)
                clean_up!(C_filt,bands[k,1],bands[k,2])
                abs_max!(C_filt)

                # fill output
                all_autocorrs[:,:,i,k] = C_filt.corr
            end

        end
    end

    return restack_times, all_autocorrs
end



function compute_stretching_dvdt(autocorrs,ref_corrs,bands,t_lims,dv_lims,ntrial,gain_correct,gain_correction=[])
    
    # handle time stuff
    t = range(0,1,size(autocorrs,1))
    t_win = findall(t->t<=t_lims[2] && t>=t_lims[1], t);

    # make output containers
    n_bands = size(bands,1)
    dvv = zeros(size(autocorrs,2),size(autocorrs,3),n_bands)
    
    # iterate through channel
    for c=1:size(autocorrs,2)

        # iterate through each frequency band
        for j=1:n_bands

            # get reference waveform (gain-corrected average autocorrelation at this channel)
            ref_s = ref_corrs[:,c,:,j]
            if gain_correct == true
                ref_s = gain_correction .* ref_s
            end
            
            # normalize and demean with respect to desired window
            ref_s = ref_s .- mean(ref_s[t_win])
            ref_s = ref_s ./ maximum(abs.(ref_s[t_win]))

            # iterate through each hour
            for i=1:size(autocorrs,3)

                # check if empty
                if sum(autocorrs[:,c,i,j]) != 0

                    # get gain-corrected waveform to compare with reference
                    s = autocorrs[:,c,i,j]
                    if gain_correct == true
                        s = s.* gain_correction
                    end
                    
                    # normalize and demean with respect to desired window
                    s = s .- mean(s[t_win])
                    s = s ./ maximum(abs.(s[t_win]))
                    
                    # calculate dv/v
                    dvv[c,i,j],cc_ts,cdp_Ts,eps_ts,err_ts,allC_ts = SeisDvv.stretching(ref_s,s,t,t_win,bands[j,1],bands[j,2],dvmin=dv_lims[1],dvmax=dv_lims[2],ntrial=ntrial);
                end
            end
        end
    end
    
    return dvv
end


function compute_stretching_dvdx(autocorrs,ref_corrs,bands,t_lims,dv_lims,ntrial,gain_correct,gain_correction=[])
    
    # handle time stuff
    t = range(0,1,size(autocorrs,1))
    t_win = findall(t->t<=t_lims[2] && t>=t_lims[1], t);

    # make output containers
    n_bands = size(bands,1)
    dvv = zeros(size(autocorrs,2),size(autocorrs,3),n_bands)
    
    # iterate through each hour
    for i=1:size(autocorrs,3)
    
        # iterate through channel
        for c=1:size(autocorrs,2)

            # iterate through each frequency band
            for j=1:n_bands

                # get reference waveform (previous channel at current hour)
                # for top channel, we will fill output with 0s since we're stretching a function with itself
                ref_s = ref_corrs[:,c,:,j]
                if gain_correct == true
                    ref_s = gain_correction .* ref_s
                end

                # normalize and demean with respect to desired window
                ref_s = ref_s .- mean(ref_s[t_win])
                ref_s = ref_s ./ maximum(abs.(ref_s[t_win]))

                # check if empty
                if sum(autocorrs[:,c,i,j]) != 0

                    # get gain-corrected waveform to compare with reference
                    s = autocorrs[:,c,i,j]
                    if gain_correct == true
                        s = s.* gain_correction
                    end
                    
                    # normalize and demean with respect to desired window
                    s = s .- mean(s[t_win])
                    s = s ./ maximum(abs.(s[t_win]))
                    
                    # calculate dv/v
                    if c == 1 || c == 1031 || c == 2061
                        dvv[c,i,j] = 0
                    else
                        dvv[c,i,j],cc_ts,cdp_Ts,eps_ts,err_ts,allC_ts = SeisDvv.stretching(ref_s,s,t,t_win,bands[j,1],bands[j,2],dvmin=dv_lims[1],dvmax=dv_lims[2],ntrial=ntrial);
                    end
                end
            end
        end
    end
    
    return dvv
end


function compute_stretching_dvdx_cumulative(autocorrs,bands,t_lims,dv_lims,ntrial,gain_correct,gain_correction=[])
    
    # handle time stuff
    t = range(0,1,size(autocorrs,1))
    t_win = findall(t->t<=t_lims[2] && t>=t_lims[1], t);

    # make output containers
    n_bands = size(bands,1)
    dvv = zeros(size(autocorrs,2),size(autocorrs,3),n_bands)
    
    # iterate through each hour
    for i=1:size(autocorrs,3)
    
        # iterate through channel
        for c=1:size(autocorrs,2)

            # iterate through each frequency band
            for j=1:n_bands

                # get reference waveform (previous channel at current hour)
                # for top channel, we will fill output with 0s since we're stretching a function with itself
                if c == 1 || c == 1031 || c == 2061
                    ref_s = autocorrs[:,1,i,j]
                else
                    ref_s = autocorrs[:,c-1,i,j]
                end
                if gain_correct == true
                    ref_s = gain_correction .* ref_s
                end

                # normalize and demean with respect to desired window
                ref_s = ref_s .- mean(ref_s[t_win])
                ref_s = ref_s ./ maximum(abs.(ref_s[t_win]))

                # check if empty
                if sum(autocorrs[:,c,i,j]) != 0

                    # get gain-corrected waveform to compare with reference
                    s = autocorrs[:,c,i,j]
                    if gain_correct == true
                        s = s.* gain_correction
                    end
                    
                    # normalize and demean with respect to desired window
                    s = s .- mean(s[t_win])
                    s = s ./ maximum(abs.(s[t_win]))
                    
                    # calculate dv/v
                    if c == 1 || c == 1031 || c == 2061
                        dvv[c,i,j] = 0
                    else
                        dvv[c,i,j],cc_ts,cdp_Ts,eps_ts,err_ts,allC_ts = SeisDvv.stretching(ref_s,s,t,t_win,bands[j,1],bands[j,2],dvmin=dv_lims[1],dvmax=dv_lims[2],ntrial=ntrial);
                    end
                end
            end
        end
    end
    
    return dvv
end


# make a helper function to get files and datetimes 
function get_files_and_datetimes(cmin,cmax,path)

    # list files
    files = glob("*.jld2",path)
    files = files[BitVector(1 .- contains.(files,"autocorrelation"))]
    files = files[BitVector(1 .- contains.(files,"datetimes"))]
    files = files[BitVector(1 .- contains.(files,"dv"))]

    # get datetimes of each file
    datetimes = []
    for i=1:size(files,1)
        datetimes = vcat(datetimes,DateTime(string("2019"*split(split(files[i],"_2019")[2],".jld2")[1])))
    end
    
    return files, path, datetimes
end
