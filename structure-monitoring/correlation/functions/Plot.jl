using SeisNoise, PyPlot, CUDA, Glob, HDF5, Combinatorics, Random, Statistics, ImageFiltering, FFTW, JLD2, Dates
import SeisNoise: NoiseData
import SeisBase: read_nodal, NodalData, InstrumentPosition, InstrumentResponse, show_str, show_t, show_x, show_os
import FFTW: rfft, irfft
import Base:show, size, summary
include("Types.jl")
include("Nodal.jl")


function plot_correlations(C,gather,maxlag,chans,source_chan,surface_chan,title_string,fname="",dpi=100,cmap="PuOr")
    if gather == "common shot"

        indices = collect(with_replacement_combinations(collect(chans[1]:chans[2]),2))
        indices = reduce(vcat,transpose.(indices))

        ssp_ind = vec(indices[:,1] .== source_chan .|| indices[:,2] .== source_chan)

        # get channel spacing
        spacing = (C.misc[4]["shot_point"]-C.misc[3]["shot_point"])/1000
        profile_start = (indices[ssp_ind,1][1] - surface_chan) * spacing
        profile_end = (indices[ssp_ind,2][end] - surface_chan) * spacing

        # make a plot of the correlations
        figure(figsize=(10,10))
        extent=[-maxlag,maxlag,profile_end,profile_start]
        imshow(C.corr[:,ssp_ind]', cmap=cmap, interpolation=:none, aspect="auto",extent=extent)
        ylabel("Distance from surface (m)",fontsize=15)
        xlabel("Lag (s)",fontsize=15)
        xticks(fontsize=12)
        yticks(fontsize=12)
        PyPlot.title(title_string)
    elseif gather == "common_midpoint"
    end
    if fname != ""
        savefig(fname,dpi=dpi)
    end
end


function plot_autocorr_time(p_autocorrs,s_autocorrs,restack_interval,start_time,skip_time,mid_chan,path)
    
    # choose some time period to skip 
    fname = string(path,"datetimes_",restack_interval,"_min_stack.jld2")
    datetimes = JLD2.load(fname)["datetimes"]
    skip_ind = [findfirst(datetimes .> skip_time[1]),findfirst(datetimes .> skip_time[2])]
    p_autocorrs[:,:,skip_ind[1]:skip_ind[2],:] .= 0
    s_autocorrs[:,:,skip_ind[1]:skip_ind[2],:] .= 0

    # get index of first correlation
    start_time = DateTime(2019,7,6,10)
    start_ind = findfirst(datetimes .> start_time)
    p_autocorrs_win = p_autocorrs[:,:,start_ind:end,:]
    s_autocorrs_win = s_autocorrs[:,:,start_ind:end,:]

    # get plot dates
    datenums = PyPlot.matplotlib.dates.date2num(datetimes)
    start_datenum = PyPlot.matplotlib.dates.date2num(start_time)

    # make plot 
    fig,ax = plt.subplots(2,2,figsize=(10,10))

    # make a gain correction function
    lambda = 0.005
    exp_func = exp.(-lambda*range(1,size(pos_p_autocorrs,1)))
    pre_trim = 150
    post_trim = pre_trim-1
    exp_func_pre = reverse(exp_func)[pre_trim:end] .- reverse(exp_func)[end]
    gain_correction = 1 .+ cat(exp_func_pre,(1 .- exp_func[1:post_trim]),dims=1)
    gain_correction = gain_correction .- gain_correction[1]

    # plot time-averaged autocorrelation from middle channel (p-waves)
    mean_p_autocorr = mean(p_autocorrs_win[:,mid_chan,(p_autocorrs_win[1,1,:,1] .!= 0),1],dims=2)
    lags = range(0,1,step=1/(size(mean_p_autocorr,1)-1))
    ax[1,1].plot(lags,mean_p_autocorr,c="k")
    ax[1,1].set_xlim(0,1)
    ax[1,1].set_ylim(-1,1)
    ax[1,1].set_xlabel("Lag time (s)")
    rect1 = PyPlot.matplotlib.patches.Rectangle([0,-1],0.3,2,color="gold",alpha=0.1)
    rect2 = PyPlot.matplotlib.patches.Rectangle([0.3,-1],0.5,2,color="royalblue",alpha=0.1)
    ax[1,1].add_patch(rect1)
    ax[1,1].add_patch(rect2)
    ax[1,1].vlines([0.3,0.8],-1,1,linestyle="--",colors="black",alpha=0.5)
    ax[1,1].text(0.05,0.9,"Ballistic",size=12)
    ax[1,1].text(0.5,0.9,"Coda",size=12)
    ax[1,1].set_title("A.       ",loc="left")

    # plot time-averaged autocorrelation from middle channel (s-waves)
    mean_s_autocorr = mean(s_autocorrs_win[:,mid_chan,(s_autocorrs_win[1,1,:,1] .!= 0),1],dims=2)
    lags = range(0,1,step=1/(size(mean_s_autocorr,1)-1))
    ax[1,2].plot(lags,mean_s_autocorr,c="k")
    ax[1,2].set_xlim(0,1)
    ax[1,2].set_ylim(-1,1)
    ax[1,2].set_xlabel("Lag time (s)")
    rect1 = PyPlot.matplotlib.patches.Rectangle([0,-1],0.3,2,color="gold",alpha=0.1)
    rect2 = PyPlot.matplotlib.patches.Rectangle([0.3,-1],0.5,2,color="royalblue",alpha=0.1)
    ax[1,2].add_patch(rect1)
    ax[1,2].add_patch(rect2)
    ax[1,2].vlines([0.3,0.8],-1,1,linestyle="--",colors="black",alpha=0.5)
    ax[1,2].text(0.05,0.9,"Ballistic",size=12)
    ax[1,2].text(0.5,0.9,"Coda",size=12)
    ax[1,2].set_title("B.        ",loc="left")

    # plot autocorrelations at middle channel through time (p-waves)
    extent = [0,1,datenums[end],start_datenum]
    corrected_p_autocorrs = gain_correction.*p_autocorrs_win[:,mid_chan,:,1]
    corrected_p_autocorrs = corrected_p_autocorrs./maximum(corrected_p_autocorrs,dims=1)
    ax[2,1].imshow(corrected_p_autocorrs',aspect="auto",cmap="seismic",
        origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[2,1].yaxis.set_major_formatter(PyPlot.matplotlib.dates.DateFormatter("%m/%d\n%H:%M"))
    ax[2,1].set_xlabel("Lag time (s)")
    ax[2,1].set_ylabel("Time")
    ax[2,1].grid(which="both")
    ax[2,1].set_title("C.",loc="left")

    # plot autocorrelations at middle channel through time (s-waves)
    corrected_s_autocorrs = gain_correction.*s_autocorrs_win[:,mid_chan,:,1]
    corrected_s_autocorrs = corrected_s_autocorrs./maximum(corrected_s_autocorrs,dims=1)
    ax[2,2].imshow(corrected_s_autocorrs',aspect="auto",cmap="seismic",
        origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[2,2].yaxis.set_major_formatter(PyPlot.matplotlib.dates.DateFormatter("%m/%d\n%H:%M"))
    ax[2,2].set_xlabel("Lag time (s)")
    ax[2,2].set_ylabel("Time")
    ax[2,2].grid(which="both")
    ax[2,2].set_title("D.",loc="left")
    plt.tight_layout()
    plt.savefig("../paper_figures/misc/autocorrelations.png",dpi=100)
end

function plot_autocorr_space(p_autocorrs,s_autocorrs,restack_interval,start_time,skip_time,path)
    
    # choose some time period to skip 
    fname = string(path,"datetimes_",restack_interval,"_min_stack.jld2")
    datetimes = JLD2.load(fname)["datetimes"]
    skip_ind = [findfirst(datetimes .> skip_time[1]),findfirst(datetimes .> skip_time[2])]
    p_autocorrs[:,:,skip_ind[1]:skip_ind[2],:] .= 0
    s_autocorrs[:,:,skip_ind[1]:skip_ind[2],:] .= 0

    # get index of first correlation
    start_time = DateTime(2019,7,6,10)
    start_ind = findfirst(datetimes .> start_time)
    p_autocorrs_win = p_autocorrs[:,:,start_ind:end,:]
    s_autocorrs_win = s_autocorrs[:,:,start_ind:end,:]

    # get plot dates
    datenums = PyPlot.matplotlib.dates.date2num(datetimes)
    start_datenum = PyPlot.matplotlib.dates.date2num(start_time)

    # make plot 
    fig,ax = plt.subplots(3,2,figsize=(10,10))

    # make a gain correction function
    lambda = 0.005
    exp_func = exp.(-lambda*range(1,size(pos_p_autocorrs,1)))
    pre_trim = 150
    post_trim = pre_trim-1
    exp_func_pre = reverse(exp_func)[pre_trim:end] .- reverse(exp_func)[end]
    gain_correction = 1 .+ cat(exp_func_pre,(1 .- exp_func[1:post_trim]),dims=1)
    gain_correction = gain_correction .- gain_correction[1]

    # plot time-averaged autocorrelation from all channels (p-waves)
    mean_p_autocorrs = mean(p_autocorrs_win[:,:,(p_autocorrs_win[1,1,:,1] .!= 0),1],dims=3)[:,:,1]
    corrected_p_autocorrs = gain_correction.*mean_p_autocorrs
    corrected_p_autocorrs = corrected_p_autocorrs./maximum(corrected_p_autocorrs,dims=1)
    extent = [0,1,1030*1.017,0]
    ax[1,1].imshow(corrected_p_autocorrs',aspect="auto",cmap="seismic",
            origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[1,1].set_title("P-wave autocorrelations with depth\n (averaged over study time period)")
    ax[1,1].set_ylabel("Depth (m)")

    # plot time-averaged autocorrelation from all channels (s-waves)
    mean_s_autocorrs = mean(s_autocorrs_win[:,:,(s_autocorrs_win[1,1,:,1] .!= 0),1],dims=3)[:,:,1]
    corrected_s_autocorrs = gain_correction.*mean_s_autocorrs
    corrected_s_autocorrs = corrected_s_autocorrs./maximum(corrected_s_autocorrs,dims=1)
    extent = [0,1,1030*1.017,0]
    ax[1,2].imshow(corrected_s_autocorrs',aspect="auto",cmap="seismic",
            origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[1,2].set_title("S-wave autocorrelations with depth\n (averaged over study time period)")

    # set daytime and nighttime indices
    night_time = DateTime(2019,7,8,17)
    night_ind = findfirst(datetimes .> night_time)
    day_time = DateTime(2019,7,8,7)
    day_ind = findfirst(datetimes .> day_time)

    # plot daytime autocorrelation from all channels (p-waves)
    p_autocorrs_day = p_autocorrs[:,:,day_ind,:]
    mean_p_autocorrs = mean(p_autocorrs_day[:,:,(p_autocorrs_day[1,1,:,1] .!= 0),1],dims=3)[:,:,1]
    corrected_p_autocorrs = gain_correction.*mean_p_autocorrs
    corrected_p_autocorrs = corrected_p_autocorrs./maximum(corrected_p_autocorrs,dims=1)
    extent = [0,1,1030*1.017,0]
    ax[2,1].imshow(corrected_p_autocorrs',aspect="auto",cmap="seismic",
            origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[2,1].set_title(string("P-wave autocorrelations with depth\n (",day_time,")"))
    ax[2,1].set_ylabel("Depth (m)")

    # plot daytime autocorrelation from all channels (s-waves)
    s_autocorrs_day = s_autocorrs[:,:,day_ind,:]
    mean_s_autocorrs = mean(s_autocorrs_day[:,:,(s_autocorrs_day[1,1,:,1] .!= 0),1],dims=3)[:,:,1]
    corrected_s_autocorrs = gain_correction.*mean_s_autocorrs
    corrected_s_autocorrs = corrected_s_autocorrs./maximum(corrected_s_autocorrs,dims=1)
    extent = [0,1,1030*1.017,0]
    ax[2,2].imshow(corrected_s_autocorrs',aspect="auto",cmap="seismic",
            origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[2,2].set_title(string("S-wave autocorrelations with depth\n (",day_time,")"))

    # plot nighttime autocorrelation from all channels (p-waves)
    p_autocorrs_night = p_autocorrs[:,:,night_ind,:]
    mean_p_autocorrs = mean(p_autocorrs_night[:,:,(p_autocorrs_night[1,1,:,1] .!= 0),1],dims=3)[:,:,1]
    corrected_p_autocorrs = gain_correction.*mean_p_autocorrs
    corrected_p_autocorrs = corrected_p_autocorrs./maximum(corrected_p_autocorrs,dims=1)
    extent = [0,1,1030*1.017,0]
    ax[3,1].imshow(corrected_p_autocorrs',aspect="auto",cmap="seismic",
            origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[3,1].set_title(string("P-wave autocorrelations with depth\n (",night_time,")"))
    ax[3,1].set_ylabel("Depth (m)")
    ax[3,1].set_xlabel("Lag (s)")

    # plot nighttime autocorrelation from all channels (s-waves)
    s_autocorrs_night = s_autocorrs[:,:,night_ind,:]
    mean_s_autocorrs = mean(s_autocorrs_night[:,:,(s_autocorrs_night[1,1,:,1] .!= 0),1],dims=3)[:,:,1]
    corrected_s_autocorrs = gain_correction.*mean_s_autocorrs
    corrected_s_autocorrs = corrected_s_autocorrs./maximum(corrected_s_autocorrs,dims=1)
    extent = [0,1,1030*1.017,0]
    ax[3,2].imshow(corrected_s_autocorrs',aspect="auto",cmap="seismic",
            origin="upper",extent=extent,interpolation="none",vmin=-1,vmax=1)
    ax[3,2].set_title(string("S-wave autocorrelations with depth\n (",night_time,")"))
    ax[3,2].set_xlabel("Lag (s)")

    plt.tight_layout()
    plt.savefig("../paper_figures/misc/autocorrelations_space.png",dpi=100)
end