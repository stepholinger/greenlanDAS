using SeisNoise, SeisIO, Glob, HDF5, Combinatorics, JLD2
include("Types.jl")
include("Nodal.jl")
include("Misc.jl")

# get list of output files
path = "/1-fnp/petasaur/p-wd03/folder/outputs/"
files = glob("*.jld2",path)

# read an output file
C = JLD2.load(files[1])["NodalCorrData"]

# set some postprocessing parameters
freqmin,freqmax = 10,20

# do some post processing using the clean_up function
clean_up!(C,freqmin,freqmax)
abs_max!(C)

# indices of resulting correlations (C.corr) are organized as below
indices = [j for j in combinations(collect(1:10),2)]
indices = reduce(vcat,transpose.(indices))