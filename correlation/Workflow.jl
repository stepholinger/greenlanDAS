using SeisNoise, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData
import SeisIO: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import Base:show, size, summary
include("Types.jl")
include("Nodal.jl")
include("Misc.jl")


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

function workflow(files,cc_len,maxlag,freqmin,freqmax,fs,cmin,cmax,sgn,time_norm,
                  chans,output_times,out_path,geometry="l",whitening=1,samples_per_file=[],mode="corr",device=3)
    
# baseline: resample > fk > detrend/taper/filter > whiten > 1bit > slice > correlate
# optimized: preprocess > rfft > fk > irfft (space) > whiten > correlate
# the preprocess function does resample/detrend/taper/filter/1bit/slice
# compromise: 
    
    # choose GPU
    device!(device)    
    
    # read the first file and collect metadata
    N = read_nodal("segy", files[1])
    if isempty(samples_per_file) == true
        samples_per_file = N.info["orignx"]
    end
    seconds_per_file = samples_per_file/N.fs[1]
    num_files = length(files)
    num_chans = chans[2]-chans[1]+1
    
    # define split point for u-shaped geometry
    split_pt = num_chans - div(num_chans,2)
    
    # make output matrix and dummy NodalFFTData
    num_corrs = 0
    if mode == "corr"
        num_corrs = Int64(num_chans*(num_chans-1)/2)+num_chans
    elseif mode == "auto"
        num_corrs = Int64(num_chans)
    elseif mode == "auto_cross"
        num_corrs = Int64((split_pt-1)*3)
    end        
    corr_mat = zeros(Int64((maxlag*fs)*2+1), num_corrs) |> cu
    corr = zeros(Int64((maxlag*fs)*2+1), num_corrs) |> cu
    
    # open file for error handling
    error_file = open(out_path*"errors.txt", "a")
    
    # iterate through each file
    next_output_ind = 1
    for i=1:num_files
        
        # exception handling
#         try

            # read the file
            N = read_nodal("segy", files[i])[chans[1]:chans[end]]
            datetime = get_datetime(N)

            # if file is incorrect length, report and go to the next one
            if size(N.data)[1] != samples_per_file || isempty(N.data) == true
                msg = "Fewer samples than expected!"
                error_string = "\nDid not correlate file "*files[i]*"\n"*msg*"\n"
                write(error_file,error_string)
                println("Did not correlate "*files[i]*" (too few samples)")
                flush(stdout)
                continue
            end

            # preprocess
            resample!(N,fs)
            detrend!(N)
            taper!(N)
            bandpass!(N,freqmin,freqmax,zerophase=true)                

            # send to GPU
            N.data = N.data |> cu

            # check which fk filtering scheme to use
            if sgn in ["both","pos","neg"]
                if geometry == "u"
                    N = apply_fk_u_geo(N,cmin,cmax,sgn,split_pt)
                else
                    N = apply_fk_l_geo(N,cmin,cmax,sgn,num_chans)
                end
            end              

            # spectral whitening
            if whitening == 1
                NF = rfft(N,[1])
                whiten!(NF,freqmin,freqmax)
                N = irfft(NF,[1])
            end

            # one bit normalization
            if time_norm == "1bit"                    
                N.data .= sign.(N.data)
            end

            # slicing
            sliced_data = slice(N,cc_len)
            NP = NodalProcessedData(N.n,size(sliced_data)[1],N.ox,N.oy,N.oz,N.info,N.id,N.name,
                       N.loc,fs*ones(N.n),N.gain,Float64(freqmin),Float64(freqmax),cc_len,"1bit",
                       N.resp,N.units,N.src,N.misc,N.notes,N.t,sliced_data)

            # cross correlate
            NF = rfft(NP,[1])
            if mode == "corr"
                corr = correlate(NF,Int64(maxlag*NF.fs[1]))
            elseif mode == "auto"
                corr = autocorrelate(NF,Int64(maxlag*NF.fs[1]))
            elseif mode == "auto_cross"
                corr = autocorrelate_cross(NF,Int64(maxlag*NF.fs[1]),split_pt)
            end
            corr_mat .= corr_mat .+ sum(corr,dims=3)
            println("Summed correlations for "*files[i])
            flush(stdout)

        # exception handling
#         catch error
#             bt = backtrace()
#             msg = sprint(showerror, error, bt)
#             error_string = "\nError on file: "*files[i]*"\n"*msg*"\n"
#             write(error_file,error_string)
#         end
        
        # get the next output time
        bf = datetime .<= output_times
        af = (datetime + Second(30)) .> output_times
        output_ind = findfirst(bf+af .== 2)
        if output_ind != nothing
            next_output_ind = output_ind + 1
        end

        # check if there's about to be a gap 
        next_datetime = datetime
        if i < num_files
            next_datetime = DateTime(split(split(files[i+1],"Q_")[2],".")[1],"yymmddHHMMSS")+Year(2000)
        end
        gap = next_datetime-datetime

        # if the gap would push us past the next output time, write a file
        if gap > Second(30) && datetime + gap > output_times[next_output_ind]
            print("Gap (",gap.value/1000/60," mins) at ",datetime," \n")
            NC = NodalCorrData(num_corrs,N.ox,N.oy,N.oz,N.info,N.id,N.name,N.loc,N.fs,
                               N.gain,Float64(freqmin),Float64(freqmax),cc_len,maxlag,"1bit",true,
                               N.resp,N.units,N.src,N.misc,N.notes,N.t,real(Array(corr_mat)))
            fname = string(out_path,"correlations_",datetime,".jld2")
            JLD2.save(fname,Dict("NodalCorrData"=>NC))
            println("Saved ",fname," (last file: "*files[i]*")")
            flush(stdout)

            # clear output matrices
            corr .= 0
            corr_mat .= 0
        end

        # if within one file duration of an output time, write a file
        if sum(bf+af .== 2) == 1
            NC = NodalCorrData(num_corrs,N.ox,N.oy,N.oz,N.info,N.id,N.name,N.loc,N.fs,
                               N.gain,Float64(freqmin),Float64(freqmax),cc_len,maxlag,"1bit",true,
                               N.resp,N.units,N.src,N.misc,N.notes,N.t,real(Array(corr_mat)))
            fname = string(out_path,"correlations_",datetime,".jld2")
            JLD2.save(fname,Dict("NodalCorrData"=>NC))
            println("Saved ",fname," (last file: "*files[i]*")")
            flush(stdout)

            # clear output matrices
            corr .= 0
            corr_mat .= 0
        end
        
    end
    close(error_file)
    return
end