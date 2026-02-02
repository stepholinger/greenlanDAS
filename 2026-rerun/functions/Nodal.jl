"""

Completed functions for updated workflow

"""

using SeisNoise, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2
import SeisNoise: NoiseData, resample!, whiten!, abs_max!, clean_up!
import SeisBase: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft, fft, ifft
import Base:show, size, summary
include("Types.jl")

# function to resample NodalData
function resample!(N::NodalData,fs::Real)
    @assert fs > 0 "New fs must be greater than 0!"
    if N.fs[1] == fs 
        return nothing 
    end
    T = eltype(N.data)
    rate = T(fs / N.fs[1])
    N.data = SeisNoise.resample_kernel(N.data, rate)
    N.fs = ones(N.n)*Float64(fs)
    return nothing
end


# slice data into time windows
function slice(N::NodalData,win_len::Int64)
    len = size(N.data)[1]
    win_len = Int64(win_len * N.fs[1])
    num_wins = len/win_len
    if isinteger(num_wins)
        sliced_data = reshape(N.data,(win_len,Int64(num_wins),N.n))
        sliced_data = permutedims(sliced_data,(1,3,2))
        return sliced_data
    else
        print("Window size must be a factor of the total data length!\n")
        return nothing
    end
end


# time domain preprocessing parent function
function preprocess(N::NodalData,freqmin::Real,freqmax::Real,fs::Real,cc_len::Real,
        time_norm::String="",chans::Vector{Int64}=[0])
    if chans != [0]
        N = N[chans[1]:chans[2]]
    end
    resample!(N,fs)
    detrend!(N)
    taper!(N)
    bandpass!(N,freqmin,freqmax,zerophase=true)
    if time_norm == "1bit" || time_norm == "onebit" || time_norm == "one bit"
        N.data .= sign.(N.data)
    # add other time normalization options (standard deviation clipping, running mean normalization)
    end
    sliced_data = slice(N,cc_len)
    return NodalProcessedData(N.n,size(sliced_data)[1],N.ox,N.oy,N.oz,N.info,N.id,N.name,
           N.loc,fs*ones(N.n),N.gain,Float64(freqmin),Float64(freqmax),cc_len,time_norm,
           N.resp,N.units,N.src,N.misc,N.notes,N.t,sliced_data)

end


function merge_channels(N1::NodalData,N2::NodalData,dim::Int64)          
    n = N1.n + N2.n
    fs = vcat(N1.fs,N2.fs)
    data = cat(N1.data,N2.data,dims=dim)
    return NodalData(n,N1.ox,N1.oy,N1.oz,N1.info,N1.id,N1.name,
               N1.loc,fs,N1.gain,N1.resp,N1.units,N1.src,N1.misc,
               N1.notes,N1.t,data)
end


# function to get the FFT metadata based on inputted dimensions
function fft_dims_flag(dims::Vector{Int64})
    if dims == [1]
        flag = "time"
    elseif dims == [2]
        flag = "space"
    elseif dims == [1,2]
        flag = "time, space"
    end
    return flag
end
            

# standard SeisNoise rfft defined for NodalData
function rfft(N::NodalData,dims::Vector{Int64}=[1])
    FFT = rfft(N.data,dims)
    ns = size(N.data)[1]
    return NodalFFTData(N.n,ns,N.ox,N.oy,N.oz,N.info,N.id,N.name,N.loc,
                        N.fs,N.gain,0.,0.,0,"N/A",N.resp,N.units,N.src,
                        N.misc,N.notes,false,dims,N.t,FFT)
end


# standard SeisNoise fft defined for NodalData
function fft(N::NodalData,dims::Vector{Int64}=[1])
    FFT = fft(N.data,dims)
    ns = size(N.data)[1]
    return NodalFFTData(N.n,ns,N.ox,N.oy,N.oz,N.info,N.id,N.name,N.loc,
                        N.fs,N.gain,0.,0.,0,"N/A",N.resp,N.units,N.src,
                        N.misc,N.notes,false,dims,N.t,FFT)
end


# standard SeisNoise rfft defined for NodalProcessedData
function rfft(NP::NodalProcessedData,dims::Vector{Int64}=[1])
    FFT = rfft(NP.data,dims)
    ns = size(NP.data)[1]
    return NodalFFTData(NP.n,ns,NP.ox,NP.oy,NP.oz,NP.info,NP.id,NP.name,
                        NP.loc,NP.fs,NP.gain,NP.freqmin,NP.freqmax,NP.cc_len,
                        NP.time_norm,NP.resp,NP.units,NP.src,NP.misc,NP.notes,
                        true,dims,NP.t,FFT)
end


# ifft for NodalFFTData
function ifft(NF::NodalFFTData,dims::Vector{Int64}=[1])
    
    # if inverting all dimensions that have been transformed, return NodalProcessedData or NodalData
    if dims == NF.dims
        data = ifft(NF.fft,dims)
        data = real(data)
        
        # check out if input was preprocessed or not, which will determined what
        # object is returned
        if NF.preprocessed
            return NodalProcessedData(NF.n,size(data)[1],NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,
                    NF.loc,NF.fs,NF.gain,NF.freqmin,NF.freqmax,NF.cc_len,NF.time_norm,
                    NF.resp,NF.units,NF.src,NF.misc,NF.notes,NF.t,data)
        else            
            return NodalData(NF.n,NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,
                    NF.loc,NF.fs,NF.gain,NF.resp,NF.units,NF.src,NF.misc,NF.notes,NF.t,data)
        end
            
    # if inverting one transform but not the other
    elseif (dims == [2] && NF.dims == [1,2]) || (dims == [1] && NF.dims == [1,2]) 
        FFT = ifft(NF.fft,dims)
        return NodalFFTData(NF.n,NF.ns,NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,
                        NF.loc,NF.fs,NF.gain,NF.freqmin,NF.freqmax,NF.cc_len,
                        NF.time_norm,NF.resp,NF.units,NF.src,NF.misc,NF.notes,
                        NF.preprocessed,[1],NF.t,FFT)

    # if inverting a non-transformed dimension
    elseif (dims == [1] && NF.dims == [2]) || (dims == [2] && NF.dims == [1]) || (dims == [1,2] && NF.dims == [1]) || (dims == [1,2] && NF.dims == [2])
        print("Cannot invert along a non-transformed dimension.\n")
        return nothing
    end
end


# ifft for NodalFFTData
function irfft(NF::NodalFFTData,dims::Vector{Int64}=[1])
    
    # if inverting all dimensions that have been transformed, return NodalProcessedData or NodalData
    if dims == NF.dims
        if dims == [1]
            data = irfft(NF.fft,NF.ns,dims)
        elseif dims == [2]
            data = irfft(NF.fft,NF.n,dims)
        elseif dims == [1,2]
            data = irfft(NF.fft,NF.ns,dims)
        end
            
        # check out if input was preprocessed or not, which will determined what
        # object is returned
        if NF.preprocessed
            return NodalProcessedData(NF.n,size(data)[1],NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,
                    NF.loc,NF.fs,NF.gain,NF.freqmin,NF.freqmax,NF.cc_len,NF.time_norm,
                    NF.resp,NF.units,NF.src,NF.misc,NF.notes,NF.t,data)
        else            
            return NodalData(NF.n,NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,
                    NF.loc,NF.fs,NF.gain,NF.resp,NF.units,NF.src,NF.misc,NF.notes,NF.t,data)
        end
            
    # if inverting space transform but not time
    elseif dims == [2] && NF.dims == [1,2]
        FFT = ifft(NF.fft,dims)
        return NodalFFTData(NF.n,NF.ns,NF.ox,NF.oy,NF.oz,NF.info,NF.id,NF.name,
                        NF.loc,NF.fs,NF.gain,NF.freqmin,NF.freqmax,NF.cc_len,
                        NF.time_norm,NF.resp,NF.units,NF.src,NF.misc,NF.notes,
                        NF.preprocessed,[1],NF.t,FFT)
    # if attempting to invert time transform but not space
    elseif dims == [1] && NF.dims == [1,2]
        print("Cannot invert along first dimension before second dimension due to RFFT algorithm design.\n")
        return nothing

    # if inverting a non-transformed dimension
    elseif (dims == [1] && NF.dims == [2]) || (dims == [2] && NF.dims == [1]) || (dims == [1,2] && NF.dims == [1]) || (dims == [1,2] && NF.dims == [2])
        print("Cannot invert along a non-transformed dimension.\n")
        return nothing
    end
end

# THIS VERSION WORKS FOR FFT, NOT RFFT
# fk filtering to remove unreasonable phase velocities
# function fk!(NF::NodalFFTData,cmin::Real,cmax::Real,sgn::String="both")
    
#     # get spacing
#     dx = (NF.misc[2]["shot_point"]-NF.misc[1]["shot_point"])/1000
    
#     # note that rfft only reduces the fft size for the first dimension, so here we must use
#     # rfftfreq for the frequencies but fftfreq (with fftshift) for wavenumbers
#     f0 = FFTW.fftshift(fftfreq(NF.ns,NF.fs[1]))
#     k0 = reverse(FFTW.fftshift(fftfreq(Int64(NF.n),1/dx)))
#     K = k0' .* ones(size(f0)[1],NF.n)
#     F = ones(size(f0)[1],NF.n) .* f0
#     C = F./K
#     filt = zeros(size(NF.fft)[1],NF.n)
#     if sgn == "both"
#         filt[abs.(C).>cmin .&& abs.(C).<cmax] .= 1.
#     elseif sgn == "pos"
#         filt[C.>cmin .&& C.<cmax] .= 1.
#     elseif sgn == "neg"
#         filt[C.<-cmin .&& C.>-cmax] .= 1.
#     end
#     filt = ImageFiltering.imfilter(Float32,filt,ImageFiltering.Kernel.gaussian(3),
#                                    "reflect",ImageFiltering.Algorithm.FIR()) 
#     if typeof(NF.fft) == CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer} || 
#         typeof(NF.fft) == CuArray{ComplexF32, 2, CUDA.Mem.DeviceBuffer} 
#         filt = filt |> cu
#     end
    
#     # apply the filter
#     NF.fft = FFTW.ifftshift(FFTW.fftshift(NF.fft).*filt)
    
#     return nothing
# end


# fk filtering to remove unreasonable phase velocities
function fk!(NF::NodalFFTData,cmin::Real,cmax::Real,sgn::String="both")
    
    # get spacing
    dx = (NF.misc[2]["shot_point"]-NF.misc[1]["shot_point"])/1000
    
    # note that rfft only reduces the fft size for the first dimension, so here we must use
    # rfftfreq for the frequencies but fftfreq (with fftshift) for wavenumbers
    f0 = rfftfreq(NF.ns,NF.fs[1])
    k0 = reverse(FFTW.fftshift(fftfreq(Int64(NF.n),1/dx)))
    K = k0' .* ones(size(f0)[1],NF.n)
    F = ones(size(f0)[1],NF.n) .* f0
    C = F./K
    filt = zeros(size(NF.fft)[1],NF.n)
    if sgn == "both"
        filt[abs.(C).>cmin .&& abs.(C).<cmax] .= 1.
    elseif sgn == "pos"
        filt[C.>cmin .&& C.<cmax] .= 1.
    elseif sgn == "neg"
        filt[C.<-cmin .&& C.>-cmax] .= 1.
    end
    filt = ImageFiltering.imfilter(Float32,filt,ImageFiltering.Kernel.gaussian(3),
                                   "reflect",ImageFiltering.Algorithm.FIR()) 
    if typeof(NF.fft) == CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer} || 
        typeof(NF.fft) == CuArray{ComplexF32, 2, CUDA.Mem.DeviceBuffer} 
        filt = filt |> cu
    end
    
    # apply the filter
    NF.fft = FFTW.ifftshift(FFTW.fftshift(NF.fft,2).*filt,2)
    
    return nothing
end



function whiten!(NF::NodalFFTData,freqmin::Real, freqmax::Real, pad::Int=50)
    @assert freqmin > 0 "Whitening frequency must be greater than zero."
    @assert freqmax <= N.fs[1] / 2 "Whitening frequency must be less than or equal to Nyquist frequency."
    num_wins = size(NF.fft,3)
    for i = 1:num_wins
        NF.fft[:,:,i] .= whiten(NF.fft[:,:,i],freqmin,freqmax,NF.fs[1],NF.ns,pad=pad)
    end
    return nothing
end

function correlate(NF::NodalFFTData,maxlag::Int)
    if length(size(NF.fft)) == 3
        Nt,Nc,Nw = size(NF.fft)
    elseif length(size(NF.fft)) == 2
        Nt,Nc = size(NF.fft)
        Nw = 1
    end
    #Ncorr = Nc * (Nc - 1) ÷ 2 
    Ncorr = Nc*(Nc-1)÷2 + Nc 
    Cout = similar(NF.fft,maxlag * 2 + 1,Ncorr,Nw)
    cstart = 0
    cend = 0
    for ii = 1:Nc
        
        # get output matrix indices
        cstart = cend + 1 
        cend = cstart + Nc - ii
        
        # reshape and multiply FFTs
        FFT1 = NF.fft[:,ii,:]
        FFT2 = NF.fft[:,ii:end,:]
        FFT1 = reshape(FFT1,Nt,1,Nw)
        corrT = irfft(conj.(FFT1) .* FFT2,NF.ns,1)

        # return corr[-maxlag:maxlag]
        t = vcat(0:Int(NF.ns / 2)-1, -Int(NF.ns / 2):-1)
        ind = findall(abs.(t) .<= maxlag)
        newind = fftshift(ind,1) 
        Cout[:,cstart:cend,:] .= corrT[newind,:,:] 
    end
    return Cout
end


function correlate_single(NF::NodalFFTData,maxlag::Int)
    if length(size(NF.fft)) == 3
        Nt,Nc,Nw = size(NF.fft)
    elseif length(size(NF.fft)) == 2
        Nt,Nc = size(NF.fft)
        Nw = 1
    end
    #Ncorr = Nc * (Nc - 1) ÷ 2 
    Ncorr = Nc
    Cout = similar(NF.fft,maxlag * 2 + 1,Ncorr,Nw)
    cstart = 0
    cend = 0
        
    # get output matrix indices
    cstart = cend + 1 
    cend = cstart + Nc - 1

    # reshape and multiply FFTs
    FFT1 = NF.fft[:,1,:]
    FFT2 = NF.fft[:,1:end,:]
    FFT1 = reshape(FFT1,Nt,1,Nw)
    corrT = irfft(conj.(FFT1) .* FFT2,NF.ns,1)

    # return corr[-maxlag:maxlag]
    t = vcat(0:Int(NF.ns / 2)-1, -Int(NF.ns / 2):-1)
    ind = findall(abs.(t) .<= maxlag)
    newind = fftshift(ind,1) 
    Cout[:,cstart:cend,:] .= corrT[newind,:,:] 
        
    return Cout
end


function autocorrelate(NF::NodalFFTData,maxlag::Int)
    if length(size(NF.fft)) == 3
        Nt,Nc,Nw = size(NF.fft)
    elseif length(size(NF.fft)) == 2
        Nt,Nc = size(NF.fft)
        Nw = 1
    end
    Ncorr = Nc 
    Cout = similar(NF.fft,maxlag * 2 + 1,Ncorr,Nw)
    cstart = 0
    cend = 0
    FFT1 = NF.fft
    FFT2 = NF.fft
    corrT = irfft(conj.(FFT1) .* FFT2,NF.ns,1)

    # return corr[-maxlag:maxlag]
    t = vcat(0:Int(NF.ns / 2)-1, -Int(NF.ns / 2):-1)
    ind = findall(abs.(t) .<= maxlag)
    newind = fftshift(ind,1) 
    Cout .= corrT[newind,:,:] 
  
    return Cout
end

function autocorrelate_cross(NF::NodalFFTData,maxlag::Int,split)
    if length(size(NF.fft)) == 3
        Nt,Nc,Nw = size(NF.fft)
    elseif length(size(NF.fft)) == 2
        Nt,Nc = size(NF.fft)
        Nw = 1
    end
    
    # get legs
    NF_leg_1 = NF.fft[:,1:split-1,:]
    NF_leg_2 = reverse(NF.fft[:,split+1:end,:],dims=2)
    Ncorr = (split-1)*3
    Cout = similar(NF.fft,maxlag * 2 + 1,Ncorr,Nw)
    
    # leg 1 auto
    corrT_leg_1 = irfft(conj.(NF_leg_1) .* NF_leg_1,NF.ns,1)

    # leg 2 auto
    corrT_leg_2 = irfft(conj.(NF_leg_2) .* NF_leg_2,NF.ns,1)

    # cros "auto"
    corrT_cross = irfft(conj.(NF_leg_1) .* NF_leg_2,NF.ns,1)

    # return corr[-maxlag:maxlag]
    t = vcat(0:Int(NF.ns / 2)-1, -Int(NF.ns / 2):-1)
    ind = findall(abs.(t) .<= maxlag)
    newind = fftshift(ind,1) 
    
    # fill output
    Cout[:,1:split-1,:] .= corrT_leg_1[newind,:,:] 
    Cout[:,split:2*split-2,:] .= corrT_leg_2[newind,:,:] 
    Cout[:,2*split-1:end,:] .= corrT_cross[newind,:,:] 

    return Cout
end

clean_up!(NC::NodalCorrData,freqmin::Real,freqmax::Real; corners::Int=4,
          zerophase::Bool=true,max_length::Real=20.) = (clean_up!(NC.corr,
          freqmin,freqmax,NC.fs[1],corners=corners,zerophase=zerophase,
          max_length=max_length);NC.freqmin=Float64(freqmin);
          NC.freqmax=Float64(freqmax);return nothing)

abs_max!(NC::NodalCorrData) = abs_max!(NC.corr)
