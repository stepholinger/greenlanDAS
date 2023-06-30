using SeisNoise, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData
import SeisIO: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import Base:show, size, summary
include("Types.jl")
include("Nodal.jl")
include("Misc.jl")


# workflow with fk filtering
function workflow(files,cc_len,maxlag,freqmin,freqmax,fs,cmin,cmax,sgn,time_norm,
                  chans,output_times,out_path,samples_per_file=[])
    
# baseline: resample > fk > detrend/taper/filter > whiten > 1bit > slice > correlate
# optimized: preprocess > rfft > fk > irfft (space) > whiten > correlate
# the preprocess function does resample/detrend/taper/filter/1bit/slice
# compromise: 
    
    # read the first file and collect metadata
    N = read_nodal("segy", files[1])
    datetime = get_datetime(N)
    if isempty(samples_per_file) == true
        samples_per_file = N.info["orignx"]
    end
    seconds_per_file = samples_per_file/N.fs[1]
    num_files = length(files)
    num_chans = chans[2]-chans[1]+1
    midpoint = Int64(chans[1]+(chans[2]+1-chans[1])/2-1)

    # make iterator for saving substacks
    t = 1
    
    # make output matrix and dummy NodalFFTData
    corr_mat = zeros(Int64((maxlag*fs)*2+1), Int64(num_chans*(num_chans-1)/2)) |> cu
    global NF = rfft(N,[1])
    
    # open file for error handling
    error_file = open(out_path*"errors.txt", "a")
    
    # iterate through each file
    for i=1:num_files
        
        # exception handling
        try
        
            # read the file
            N = read_nodal("segy", files[i])[chans[1]:chans[end]]

                # check that file is correct length
                if N.info["orignx"] == samples_per_file && isempty(N.data) == false
                    # preprocess
                    resample!(N,fs)
                    detrend!(N)
                    taper!(N)
                    bandpass!(N,freqmin,freqmax,zerophase=true)

                    # send to GPU
                    N.data = N.data |> cu

                    # apply fk filter to each leg of the cable
                    split = midpoint-chans[1]+1
                    N_leg_1 = N[1:split]
                    N_leg_2 = N[split+1:end]
                    NF_leg_1 = rfft(N_leg_1,[1,2])
                    NF_leg_2 = rfft(N_leg_2,[1,2])
                    fk!(NF_leg_1,cmin,cmax,sgn)
                    fk!(NF_leg_2,cmin,cmax,sgn)
                    N_leg_1 = irfft(NF_leg_1,[1,2])
                    N_leg_2 = irfft(NF_leg_2,[1,2])
                    N = merge_channels(N_leg_1,N_leg_2,2)

                    # spectral whitening- probably can group fk and whitening to remove one fft
                    # need to get updated whitening code up to snuff
                    NF = rfft(N,[1])
                    whiten!(NF,freqmin,freqmax)
                    N = irfft(NF,[1])

                    # one bit normalization
                    N.data .= sign.(N.data)

                    # slice
                    sliced_data = slice(N,10)
                    NP = NodalProcessedData(N.n,size(sliced_data)[1],N.ox,N.oy,N.oz,N.info,N.id,N.name,
                               N.loc,fs*ones(N.n),N.gain,Float64(freqmin),Float64(freqmax),cc_len,"1bit",
                               N.resp,N.units,N.src,N.misc,N.notes,N.t,sliced_data)

                    # cross correlate- CHECKED
                    NF = rfft(NP,[1])
                    corr = correlate(NF,Int64(maxlag*NF.fs[1]))
                    corr_mat = corr_mat + sum(corr,dims=3)
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
            datetime = datetime + Second(seconds_per_file)
        end       
        if datetime <= output_times[t] && datetime+Second(seconds_per_file) > output_times[t]
            t = t + 1
            corr_mat = real(reshape(Array(corr_mat),size(corr_mat,1),size(corr_mat,2)))
            NC = NodalCorrData(NF.n,NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,NF.loc,NF.fs,
                               NF.gain,NF.freqmin,NF.freqmax,cc_len,maxlag,"1bit",true,NF.resp,NF.units,
                               NF.src,NF.misc,NF.notes,NF.t,corr_mat)
            fname = string(out_path,"correlations_",datetime,".jld2")
            JLD2.save(fname,Dict("NodalCorrData"=>NC))

            # clear output matrix
            corr_mat = zeros(Int64((maxlag*fs)*2+1), Int64(num_chans*(num_chans-1)/2)) |> cu
        end
        
    end
    close(error_file)
    return
end



# workflow without fk filtering
function workflow(files,cc_len,maxlag,freqmin,freqmax,fs,time_norm,
                  chans,output_times,out_path,samples_per_file=[])
    
# baseline: resample > fk > detrend/taper/filter > whiten > 1bit > slice > correlate
# optimized: preprocess > rfft > fk > irfft (space) > whiten > correlate
# the preprocess function does resample/detrend/taper/filter/1bit/slice
# compromise: 
    
    # read the first file and collect metadata
    N = read_nodal("segy", files[1])
    datetime = get_datetime(N)
    if isempty(samples_per_file) == true
        samples_per_file = N.info["orignx"]
    end
    seconds_per_file = samples_per_file/N.fs[1]
    num_files = length(files)
    num_chans = chans[2]-chans[1]+1
    midpoint = Int64(chans[1]+(chans[2]+1-chans[1])/2-1)

    # make iterator for saving substacks
    t = 1
    
    # make output matrix and dummy NodalFFTData
    corr_mat = zeros(Int64((maxlag*fs)*2+1), Int64(num_chans*(num_chans-1)/2)) |> cu
    global NF = rfft(N,[1])
    
    # open file for error handling
    error_file = open(out_path*"errors.txt", "a")
    
    # iterate through each file
    for i=1:num_files
        
        # exception handling
        try
        
            # read the file
            N = read_nodal("segy", files[i])[chans[1]:chans[end]]

                # check that file is correct length
                if N.info["orignx"] == samples_per_file && isempty(N.data) == false
                    # preprocess
                    resample!(N,fs)
                    detrend!(N)
                    taper!(N)
                    bandpass!(N,freqmin,freqmax,zerophase=true)

                    # send to GPU
                    N.data = N.data |> cu

                    # spectral whitening- probably can group fk and whitening to remove one fft
                    # need to get updated whitening code up to snuff
                    NF = rfft(N,[1])
                    whiten!(NF,freqmin,freqmax)
                    N = irfft(NF,[1])

                    # one bit normalization
                    N.data .= sign.(N.data)

                    # slice
                    sliced_data = slice(N,10)
                    NP = NodalProcessedData(N.n,size(sliced_data)[1],N.ox,N.oy,N.oz,N.info,N.id,N.name,
                               N.loc,fs*ones(N.n),N.gain,Float64(freqmin),Float64(freqmax),cc_len,"1bit",
                               N.resp,N.units,N.src,N.misc,N.notes,N.t,sliced_data)

                    # cross correlate- CHECKED
                    NF = rfft(NP,[1])
                    corr = correlate(NF,Int64(maxlag*NF.fs[1]))
                    corr_mat = corr_mat + sum(corr,dims=3)
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
            datetime = datetime + Second(seconds_per_file)
        end       
        if datetime <= output_times[t] && datetime+Second(seconds_per_file) > output_times[t]
            t = t + 1
            corr_mat = real(reshape(Array(corr_mat),size(corr_mat,1),size(corr_mat,2)))
            NC = NodalCorrData(NF.n,NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,NF.loc,NF.fs,
                               NF.gain,NF.freqmin,NF.freqmax,cc_len,maxlag,"1bit",true,NF.resp,NF.units,
                               NF.src,NF.misc,NF.notes,NF.t,corr_mat)
            fname = string(out_path,"correlations_",datetime,".jld2")
            JLD2.save(fname,Dict("NodalCorrData"=>NC))

            # clear output matrix
            corr_mat = zeros(Int64((maxlag*fs)*2+1), Int64(num_chans*(num_chans-1)/2)) |> cu
        end
        
    end
    close(error_file)
    return
end



# # optimized workflow (to use, paste in inner loop of workflow function)

# # time-domain preprocessing
# NP = preprocess(N,freqmin,freqmax,fs,cc_len,time_norm)

# # fk filter
# NF = rfft(NP,[1,2])
# fk!(NF,cmin,cmax,sgn)
# NF = irfft(NF,[2])

# # spectral whitening- probably can group fk and whitening to remove one fft
# # need to get updated whitening code up to snuff
# whiten!(NF,freqmin,freqmax)

# # cross correlate- CHECKED
# NF.fft = NF.fft |> cu
# corr = correlate(NF,Int64(maxlag*NF.fs[1]))
# corr_mat = corr_mat + sum(corr,dims=3)