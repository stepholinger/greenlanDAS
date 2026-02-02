"""

Define necessary structures for handling DAS data with SeisNoise functionality 

"""

using SeisNoise, SeisBase, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2
import SeisNoise: NoiseData
import SeisBase: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import Base:show, size, summary

mutable struct NodalFFTData <: NoiseData
    n::Int64
    ns::Int64
    ox::Float64                         # origin x
    oy::Float64                         # origin y
    oz::Float64                         # origin z
    info::Dict{String,Any}              # info
    id::Array{String,1}                 # id
    name::Array{String,1}               # name
    loc::Array{InstrumentPosition,1}    # loc
    fs::Array{Float64,1}                # fs
    gain::Array{Float64,1}              # gain
    freqmin::Float64                    # minumum frequency [Hz]
    freqmax::Float64                    # maximum frequency [Hz]
    cc_len::Int64                       # window_length [S]
    time_norm::String                   # time normaliation
    resp::Array{InstrumentResponse,1}   # resp
    units::Array{String,1}              # units
    src::Array{String,1}                # src
    misc::Array{Dict{String,Any},1}     # misc
    notes::Array{Array{String,1},1}     # notes
    preprocessed::Bool
    dims::Array{Int64,1}                # FFT dimensions
    t::Array{Array{Int64,2},1}          # time
    fft::AbstractArray{<:Complex{<:AbstractFloat}}  # fft data

    function NodalFFTData(     
        n::Int64,
        ns::Int64,
        ox::Float64,                         # origin x
        oy::Float64,                         # origin y
        oz::Float64,                         # origin z
        info::Dict{String,Any},              # info
        id::Array{String,1},                 # id
        name::Array{String,1},               # name
        loc::Array{InstrumentPosition,1},    # loc
        fs::Array{Float64,1},                # fs
        gain::Array{Float64,1},              # gain
        freqmin::Float64,                    # minumum frequency [Hz]
        freqmax::Float64,                    # maximum frequency [Hz]
        cc_len::Int64,                       # window_length [S]
        time_norm::String,                   # time normaliation
        resp::Array{InstrumentResponse,1},   # resp
        units::Array{String,1},              # units
        src::Array{String,1},                # src
        misc::Array{Dict{String,Any},1},     # misc
        notes::Array{Array{String,1},1},     # notes
        preprocessed::Bool,
        dims::Array{Int64,1},                # FFT dimensions
        t::Array{Array{Int64,2},1},          # time
        fft::AbstractArray{<:Complex{<:AbstractFloat}}  # fft data
        )
        
        return new(n,ns,ox,oy,oz,info,id,name,loc,fs,gain,freqmin,freqmax,cc_len,
                   time_norm,resp,units,src,misc,notes,preprocessed,dims,t,fft)
    end
    
end


function show(io::IO, NF::NodalFFTData)
    W = max(80, displaysize(io)[2]) - show_os
    nc = getfield(NF, :n)
    w = min(W, 35)
    N = min(nc, div(W-1, w))
    M = min(N+1, nc)
    println(io, "NodalFFTData with ", nc, " channels (", N, " shown)")
    F = fieldnames(NodalFFTData)
    for f in F
        if ((f in unindexed_fields) == false) || (f == :x)
            targ = getfield(NF, f)
            t = typeof(targ)
            fstr = uppercase(String(f))
            print(io, lpad(fstr, show_os-2), ": ")
            if t == Array{String,1}
                show_str(io, targ, w, N)
            elseif f == :notes || f == :misc
                show_str(io, String[string(length(getindex(targ, i)), " entries") for i = 1:M], w, N)
            elseif f == :t           
                show_t(io, targ, w, N, NF.fs)
            elseif f == :x
                x_str = mkxstr(N, getfield(NF, :x))
                show_x(io, x_str, w, N<nc)
            else
                show_str(io, String[repr("text/plain", targ[i], context=:compact=>true) for i = 1:M], w, N)
            end
        elseif f == :ox
            print(io, "COORDS: X = ", repr("text/plain", getfield(NF, f), context=:compact=>true), ", ")
        elseif f == :oy
            print(io, "Y = ", repr("text/plain", getfield(NF, f), context=:compact=>true), ", ")
        elseif f == :oz
            print(io, "Z = ", repr("text/plain", getfield(NF, f), context=:compact=>true), "\n")
        elseif f == :info
            print(io, "  INFO: ", length(NF.info), " entries\n")
        elseif f == :dims
            dims_flag = fft_dims_flag(NF.dims)
            print(io, "  DIMS: ", dims_flag, "\n")
        end
    end
    return nothing
end
show(F::NodalFFTData) = show(stdout, F)

"""

Define necessary structures for handling sliced DAS data with SeisNoise functionality 

"""

mutable struct NodalProcessedData <: NoiseData
    n::Int64
    ns::Int64
    ox::Float64                         # origin x
    oy::Float64                         # origin y
    oz::Float64                         # origin z
    info::Dict{String,Any}              # info
    id::Array{String,1}                 # id
    name::Array{String,1}               # name
    loc::Array{InstrumentPosition,1}    # loc
    fs::Array{Float64,1}                # fs
    gain::Array{Float64,1}              # gain
    freqmin::Float64                    # minumum frequency [Hz]
    freqmax::Float64                    # maximum frequency [Hz]
    cc_len::Int64                       # window_length [S]
    time_norm::String                   # time normaliation
    resp::Array{InstrumentResponse,1}   # resp
    units::Array{String,1}              # units
    src::Array{String,1}                # src
    misc::Array{Dict{String,Any},1}     # misc
    notes::Array{Array{String,1},1}     # notes
    t::Array{Array{Int64,2},1}          # time
    data::AbstractArray{Float32, 3}     # fft data

    function NodalProcessedData(     
        n::Int64,
        ns::Int64,
        ox::Float64,                         # origin x
        oy::Float64,                         # origin y
        oz::Float64,                         # origin z
        info::Dict{String,Any},              # info
        id::Array{String,1},                 # id
        name::Array{String,1},               # name
        loc::Array{InstrumentPosition,1},    # loc
        fs::Array{Float64,1},                # fs
        gain::Array{Float64,1},              # gain
        freqmin::Float64,                    # minumum frequency [Hz]
        freqmax::Float64,                    # maximum frequency [Hz]
        cc_len::Int64,                      # window_length [S]
        time_norm::String,                   # time normaliation
        resp::Array{InstrumentResponse,1},   # resp
        units::Array{String,1},              # units
        src::Array{String,1},                # src
        misc::Array{Dict{String,Any},1},     # misc
        notes::Array{Array{String,1},1},     # notes
        t::Array{Array{Int64,2},1},          # time
        data::AbstractArray{Float32, 3}      # fft data
        )
        
        return new(n,ns,ox,oy,oz,info,id,name,loc,fs,gain,
                   freqmin,freqmax,cc_len,time_norm,resp,
                   units,src,misc,notes,t,data)
    end
    
end

function show(io::IO, NP::NodalProcessedData)
    W = max(80, displaysize(io)[2]) - show_os
    nc = getfield(NP, :n)
    w = min(W, 35)
    N = min(nc, div(W-1, w))
    M = min(N+1, nc)
    print()
    println(io, "NodalProcessedData with ", nc, " channels (", N, " shown)")
    F = fieldnames(NodalProcessedData)
    for f in F
        if ((f in unindexed_fields) == false) || (f == :x)
            targ = getfield(NP, f)
            t = typeof(targ)
            fstr = uppercase(String(f))
            print(io, lpad(fstr, show_os-2), ": ")
            if t == Array{String,1}
                show_str(io, targ, w, N)
            elseif f == :notes || f == :misc
                show_str(io, String[string(length(getindex(targ, i)), " entries") for i = 1:M], w, N)
            elseif f == :t           
                show_t(io, targ, w, N, NP.fs)
            elseif f == :x
                x_str = mkxstr(N, getfield(NP, :x))
                show_x(io, x_str, w, N<nc)
            else
                show_str(io, String[repr("text/plain", targ[i], context=:compact=>true) for i = 1:M], w, N)
            end
        elseif f == :ox
            print(io, "COORDS: X = ", repr("text/plain", getfield(NP, f), context=:compact=>true), ", ")
        elseif f == :oy
            print(io, "Y = ", repr("text/plain", getfield(NP, f), context=:compact=>true), ", ")
        elseif f == :oz
            print(io, "Z = ", repr("text/plain", getfield(NP, f), context=:compact=>true), "\n")
        elseif f == :info
            print(io, "  INFO: ", length(NP.info), " entries\n")
        end
    end
    return nothing
end
show(NP::NodalProcessedData) = show(stdout, NP)

"""

Define necessary structures for handling DAS data with SeisNoise functionality 

"""

mutable struct NodalCorrData <: NoiseData
    n::Int64
    ox::Float64                         # origin x
    oy::Float64                         # origin y
    oz::Float64                         # origin z
    info::Dict{String,Any}              # info
    id::Array{String,1}                 # id
    name::Array{String,1}               # name
    loc::Array{InstrumentPosition,1}    # loc
    fs::Array{Float64,1}                # fs
    gain::Array{Float64,1}              # gain
    freqmin::Float64                    # minumum frequency [Hz]
    freqmax::Float64                    # maximum frequency [Hz]
    cc_len::Int64                       # window_length [S]
    maxlag::Int64
    time_norm::String                   # time normaliation
    whitened::Bool
    resp::Array{InstrumentResponse,1}   # resp
    units::Array{String,1}              # units
    src::Array{String,1}                # src
    misc::Array{Dict{String,Any},1}     # misc
    notes::Array{Array{String,1},1}     # notes
    t::Array{Array{Int64,2},1}          # time
    corr::AbstractArray{<:AbstractFloat,2}  # fft data

    function NodalCorrData(     
        n::Int64,
        ox::Float64,                         # origin x
        oy::Float64,                         # origin y
        oz::Float64,                         # origin z
        info::Dict{String,Any},              # info
        id::Array{String,1},                 # id
        name::Array{String,1},               # name
        loc::Array{InstrumentPosition,1},    # loc
        fs::Array{Float64,1},                # fs
        gain::Array{Float64,1},              # gain
        freqmin::Float64,                    # minumum frequency [Hz]
        freqmax::Float64,                    # maximum frequency [Hz]
        cc_len::Int64,                       # window_length [S]
        maxlag::Int64,                       # window_length [S]
        time_norm::String,                   # time normaliation
        whitened::Bool,
        resp::Array{InstrumentResponse,1},   # resp
        units::Array{String,1},              # units
        src::Array{String,1},                # src
        misc::Array{Dict{String,Any},1},     # misc
        notes::Array{Array{String,1},1},     # notes
        t::Array{Array{Int64,2},1},          # time
        corr::AbstractArray{<:AbstractFloat,2}  # fft data
        )
        
        return new(n,ox,oy,oz,info,id,name,loc,fs,gain,freqmin,freqmax,cc_len,
                   maxlag,time_norm,whitened,resp,units,src,misc,notes,t,corr)
    end
    
end

const unindexed_fields = (:n, :ns, :ox, :oy, :oz, :freqmax ,:freqmin, :cc_len, :maxlag, :whitened,
                          :time_norm, :data, :info, :x, :preprocessed, :dims)

function show(io::IO, NC::NodalCorrData)
    W = max(80, displaysize(io)[2]) - show_os
    nc = getfield(NC, :n)
    w = min(W, 35)
    N = min(nc, div(W-1, w))
    M = min(N+1, nc)
    println(io, "NodalCorrData with ", nc, " correlations (", N, " shown)")
    F = fieldnames(NodalCorrData)
    for f in F
        if ((f in unindexed_fields) == false) || (f == :x)
            targ = getfield(NC, f)
            t = typeof(targ)
            fstr = uppercase(String(f))
            print(io, lpad(fstr, show_os-2), ": ")
            if t == Array{String,1}
                show_str(io, targ, w, N)
            elseif f == :notes || f == :misc
                show_str(io, String[string(length(getindex(targ, i)), " entries") for i = 1:M], w, N)
            elseif f == :t           
                show_t(io, targ, w, N, NC.fs)
            elseif f == :x
                x_str = mkxstr(N, getfield(NC, :x))
                show_x(io, x_str, w, N<nc)
            else
                show_str(io, String[repr("text/plain", targ[i], context=:compact=>true) for i = 1:M], w, N)
            end
        elseif f == :ox
            print(io, "COORDS: X = ", repr("text/plain", getfield(NC, f), context=:compact=>true), ", ")
        elseif f == :oy
            print(io, "Y = ", repr("text/plain", getfield(NC, f), context=:compact=>true), ", ")
        elseif f == :oz
            print(io, "Z = ", repr("text/plain", getfield(NC, f), context=:compact=>true), "\n")
        elseif f == :info
            print(io, "  INFO: ", length(NC.info), " entries\n")
        end
    end
    return nothing
end
show(NC::NodalCorrData) = show(stdout, NC)