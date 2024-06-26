using SeisNoise, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData
import SeisBase: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import Base:show, size, summary
include("Types.jl")
include("Nodal.jl")

function get_datetime(N::NodalData)
    date = [N.misc[1]["year"],N.misc[1]["day"],N.misc[1]["hour"],N.misc[1]["minute"],N.misc[1]["second"]]
    datetime = DateTime(date[1])+Day(date[2]-1)+Hour(date[3])+Minute(date[4])+Second(date[5])
    return datetime
end

# the indices for this are currently set up to work properly for chans=[331,2391]. may not work for even set of channels!
# if there's not a "middle channel", this may need to be updated. If statements?
function cross_cable_stack(C,chans)

    n = chans[end] - (chans[1])
    midpoint = (chans[1]) + Int64(n/2)
    #midpoint = (chans[1]-1) + Int64(n/2)

    # get indices for desired channels
    #indices = [j for j in combinations(chans,2)]
    indices = collect(with_replacement_combinations(chans,2))
    indices = reduce(vcat,transpose.(indices))

    # get correlation functions
    corr = C.corr

    # leg one pairs- organize them all so surface is top of array TO DO sort by distance from surface? 
    leg1_indices = (indices[:,1] .<= midpoint .&& indices[:,2] .<= midpoint)
    C_leg1 = corr[:,leg1_indices]

    # leg two pairs
    i = sortperm(indices[:,2])
    sorted_indices = reverse(indices[i,:])
    sorted_corr = reverse(corr[:,i])
    leg2_indices = (sorted_indices[:,1] .>= midpoint .&& sorted_indices[:,2] .>= midpoint)
    #leg2_indices = (sorted_indices[:,1] .> midpoint .&& sorted_indices[:,2] .> midpoint)
    C_leg2 = sorted_corr[:,leg2_indices]

    # leg three pairs
    i = sortperm(indices[:,2])
    sorted_indices2 = reverse(indices[i,:])
    sorted_corr2 = reverse(corr[:,i])
    i = sortperm(sorted_indices2[:,2])
    sorted_indices2 = reverse(sorted_indices2[i,:],dims=2)
    sorted_corr2 = reverse(sorted_corr2[:,i],dims=1)
    cross_indices = (sorted_indices2[:,1] .<= midpoint .&& sorted_indices2[:,2] .>= midpoint)
    #cross_indices = (sorted_indices2[:,1] .<= midpoint .&& sorted_indices2[:,2] .> midpoint)
    leg3_indices = (cross_indices + (reduce(vcat,sum(sorted_indices2,dims=2)) .<= chans[end]+chans[1]-1) .== 2)
    C_leg3 = sorted_corr2[:,leg3_indices]

    # leg four pairs
    i = sortperm(indices[:,1])
    sorted_indices3 = reverse(indices[i,:])
    sorted_corr3 = reverse(corr[:,i])
    i = sortperm(sorted_indices3[:,1])
    sorted_indices3 = reverse(sorted_indices3[i,:])
    sorted_corr3 = reverse(sorted_corr3[:,i])
    cross_indices = (sorted_indices3[:,1] .<= midpoint .&& sorted_indices3[:,2] .> midpoint)
    leg4_indices = (cross_indices + (reduce(vcat,sum(sorted_indices3,dims=2)) .> chans[end]+chans[1]) .== 2)
    C_leg4 = reverse(sorted_corr3[:,leg4_indices],dims=1)

    return C_leg1+C_leg2#+C_leg3+C_leg4
end


function compute_rms(files,freq,filt_type,fs,chans,out_path,samples_per_file=[],files_per_save=1e10)
    
    # read the first file and collect metadata
    N = read_nodal("segy", files[1])
    datetime = get_datetime(N)
    if isempty(samples_per_file) == true
        samples_per_file = N.info["orignx"]
    end
    seconds_per_file = samples_per_file/N.fs[1]
    num_files = length(files)
    num_chans = chans[2]-chans[1]+1

    # make output matrix and dummy NodalFFTData
    rms_mat = zeros(Int64(num_chans),Int64(size(files,1)))
    t = DateTime[]
    t = cat(t,datetime,dims=1)
    
    # open file for error handling and output
    error_file = open(out_path*"errors.txt", "a")

    # iterate through each file
    for i=1:num_files
        
        # exception handling
        try
        
            # read the file
            N = read_nodal("segy", files[i])[chans[1]:chans[end]]
                
                # check that file is correct length
                if size(N.data,1) == samples_per_file && isempty(N.data) == false
                    # preprocess
                    resample!(N,fs)
                    detrend!(N)
                    taper!(N)
                    if filt_type == "bandpass"
                        bandpass!(N,freq[1],freq[2],zerophase=true)
                    elseif filt_type == "lowpass"
                        lowpass!(N,freq,zerophase=true)
                    elseif filt_type == "highpass"
                        highpass!(N,freq[2],zerophase=true)
                    end
                
                    # compute rms
                    rms = sqrt.(sum(N.data.*N.data,dims=1)/size(N.data,1))
                    rms_mat[:,i] = rms
                
                end
            
        # exception handling
        catch error
            bt = backtrace()
            msg = sprint(showerror, error, bt)
            error_string = "\nError on file: "*files[i]*"\n"*msg*"\n"
            write(error_file,error_string)
        end
        
        # count time steps and save output
        if i > 1
            sec = Second(floor(seconds_per_file))+Millisecond(1000*(round(seconds_per_file - floor(seconds_per_file),digits=2)))
            datetime = datetime + sec
            t = cat(t,datetime,dims=1)
        end          
        
        # incremental saving 
        if mod(i,files_per_save) == 0
            JLD2.save(out_path*"rms.jld2", "rms", rms_mat)
            JLD2.save(out_path*"t.jld2", "t", t)
        end
    end
    
    
    # write final output
    JLD2.save(out_path*"rms.jld2", "rms", rms_mat)
    JLD2.save(out_path*"t.jld2", "t", t)
    close(error_file)

    return rms_mat,t
end


function apply_fk_u_geo(N,cmin,cmax,sgn,split_pt)
    
    # split into each leg
    N_leg_1 = N[1:split_pt]
    N_leg_2 = N[split_pt+1:end]

    # pad with zeros along spatial axis to avoid wrapping issues
    # not needed for temporal axis, which has been tapered already
    pad = zeros(size(N_leg_1.data,1),100)|>cu
    N_leg_1.data = hcat(pad,N_leg_1.data,pad)
    N_leg_2.data = hcat(pad,N_leg_2.data,pad)
    N_leg_1.n = size(N_leg_1.data,2)
    N_leg_2.n = size(N_leg_2.data,2)

    # take fft for each leg, fk filter, and take ifft
    NF_leg_1 = rfft(N_leg_1,[1,2])
    NF_leg_2 = rfft(N_leg_2,[1,2])
    if sgn == "both"
        fk!(NF_leg_1,cmin,cmax,sgn)
        fk!(NF_leg_2,cmin,cmax,sgn)
    elseif sgn == "pos"
        fk!(NF_leg_1,cmin,cmax,sgn)
        fk!(NF_leg_2,cmin,cmax,"neg")
    elseif sgn == "neg"
        fk!(NF_leg_1,cmin,cmax,sgn)
        fk!(NF_leg_2,cmin,cmax,"pos")
    end
    N_leg_1 = irfft(NF_leg_1,[1,2])
    N_leg_2 = irfft(NF_leg_2,[1,2])

    # remove padding and merge
    pad_bound = size(pad,2)
    N_leg_1.data = N_leg_1.data[:,pad_bound+1:pad_bound+split_pt]
    N_leg_2.data = N_leg_2.data[:,pad_bound+1:pad_bound+split_pt-1]
    N_leg_1.n = split_pt
    N_leg_2.n = split_pt-1
    N = merge_channels(N_leg_1,N_leg_2,2)
    return N
end

function apply_fk_l_geo(N,cmin,cmax,sgn,num_chans)
    
    # pad with zeros
    pad = zeros(size(N.data,1),100)|>cu
    N.data = hcat(pad,N.data,pad)
    N.n = size(N.data,2)

    # take fft, fk filter, and take ifft
    NF = rfft(N,[1,2])
    fk!(NF,cmin,cmax,sgn)
    N = irfft(NF,[1,2])

    # remove padding
    pad_bound = size(pad,2)                       
    N.data = N.data[:,pad_bound+1:pad_bound+num_chans]
    N.n = num_chans
    return N
end