# Structure Monitoring

This folder contains workflows for monitoring glacier structure using ambient noise correlation, velocity change analysis, and seismic imaging techniques.

## Contents

- `correlation/` - Main ambient noise correlation framework (Julia, GPU-accelerated)
  - Core workflow scripts
  - Batch processing scripts for different wave types
  - Analysis and visualization notebooks
  - Custom functions and data types
  
- `Olinger_etal_2026_scripts/` - Publication-ready analysis scripts
  - Autocorrelation and velocity change analysis
  - Surface wave cross-correlation
  - Study site visualization
  - Ice flow modeling

## Overview

Ambient noise correlation is used to:
- Extract coherent seismic signals from random noise
- Monitor temporal changes in seismic velocity (dv/v)
- Image subsurface structure of the glacier
- Detect and track velocity changes related to environmental conditions

## Correlation Workflow (`correlation/`)

### Main Scripts

**`run_correlation.jl`** - Template script for running correlations
- Configure parameters for your analysis
- Run full correlation workflow
- Customizable for different wave types and frequency bands

**Batch Processing Scripts** (`batch/`):
- `run_surface_correlation.jl` - Surface wave correlations with FK filtering
- `run_surface_correlation_no_fk.jl` - Surface waves without FK filter
- `run_correlation_p.jl` - P-wave correlations (high apparent velocity)
- `run_correlation_s.jl` - S-wave correlations (intermediate velocity)

### Core Functions (`functions/`)

- **`Workflow.jl`** - Main correlation pipeline
  - `preprocess_raw!()` - Detrend, taper, resample
  - `apply_fk_filter!()` - Apparent velocity filtering
  - `spectral_whitening!()` - Frequency domain whitening
  - `time_normalize!()` - 1-bit or running absolute mean normalization
  - `correlate!()` - Cross-correlation computation
  - `stack!()` - Linear or phase-weighted stacking

- **`Types.jl`** - Custom data structures
  - `NodalFFTData` - FFT-ready preprocessed data
  - `NodalProcessedData` - Fully processed data ready for correlation
  - `NodalCorrData` - Correlation results

- **`Nodal.jl`** - DAS-specific utilities
  - `read_nodal_segy()` - Read DAS SEGY files
  - `resample_nodal()` - Efficient resampling
  - `fft_nodal()` - GPU-accelerated FFT

- **`Dvv.jl`** - Velocity change analysis
  - `stretching_method()` - Compute dv/v using stretching
  - `mwcs()` - Moving window cross-spectral analysis
  - `compute_dvv_stack()` - Stack and compute dv/v time series

- **`Plot.jl`** - Visualization functions
  - `plot_correlations()` - Plot correlation matrices
  - `plot_dvv()` - Plot velocity change time series
  - `plot_moveout()` - Plot correlation moveout

- **`Misc.jl`** - Utility functions
  - `get_datetime()` - Parse DAS file timestamps
  - `compute_rms()` - RMS computation
  - `cross_cable_stack()` - Stack across cable geometry

### Analysis Notebooks

- **`autocorrelation_visualization.ipynb`** - Visualize autocorrelation results
- **`postprocessing.ipynb`** - Filter, stack, and analyze correlations
- **`misc.ipynb`** - Miscellaneous analysis and exploration
- **`debug.ipynb`** - Debugging correlation workflows

## Usage

### Basic Correlation Workflow

```bash
cd structure-monitoring/correlation
julia --project=../.. run_correlation.jl
```

**Edit `run_correlation.jl` to configure**:
```julia
# Time parameters
starttime = DateTime(2022, 7, 15)
endtime = DateTime(2022, 7, 20)

# Channel parameters
channels = 331:2391  # Full cable
# channels = 331:1361  # Surface section only

# Frequency band
freqmin = 2.0  # Hz
freqmax = 8.0  # Hz

# FK filter (for surface waves)
cmin = 100.0   # m/s (minimum apparent velocity)
cmax = 500.0   # m/s (maximum apparent velocity)

# Correlation parameters
maxlag = 10.0  # seconds
cc_step = 1800 # seconds between correlations
cc_len = 3600  # correlation window length (seconds)

# Time normalization
time_norm = "1bit"  # Options: "1bit", "ram" (running absolute mean)

# Output
output_dir = "correlations_2-8Hz_surface/"
```

### Batch Processing Different Wave Types

**Surface Waves** (low apparent velocity, 100-500 m/s):
```bash
cd structure-monitoring/correlation/batch
julia --project=../../.. run_surface_correlation.jl
```

**P-Waves** (high apparent velocity, > 3000 m/s):
```bash
julia --project=../../.. run_correlation_p.jl
```

**S-Waves** (intermediate velocity, 1500-3000 m/s):
```bash
julia --project=../../.. run_correlation_s.jl
```

### Post-Processing

```bash
cd structure-monitoring/correlation
jupyter notebook postprocessing.ipynb
```

**Common post-processing tasks**:
1. Load correlation results (JLD2 files)
2. Stack correlations over time
3. Apply additional filtering
4. Compute velocity changes (dv/v)
5. Generate figures for publication

### Velocity Change Analysis

```julia
using JLD2
using Statistics

# Load autocorrelations
corr = load("autocorrelations.jld2")

# Compute dv/v using stretching method
dvv, cc, cdp = stretching_method(
    corr["data"],
    corr["time"],
    freqmin=2.0,
    freqmax=8.0,
    dvv_range=-0.05:0.0001:0.05
)

# Plot results
plot_dvv(corr["time"], dvv, cc)
```

## Key Processing Steps

### 1. Preprocessing
```
Raw SEGY → Detrend → Taper → Resample → Bandpass Filter
```

### 2. FK Filtering (Optional)
```
FFT (t,x) → 2D FFT (f,k) → Filter by apparent velocity → Inverse FFT
```
- Separates waves by propagation velocity
- Removes unwanted wave types
- Critical for surface wave extraction

### 3. Spectral Whitening
```
FFT → Normalize amplitude spectrum → Inverse FFT
```
- Broadens frequency content
- Equalizes different frequency contributions
- Improves correlation quality

### 4. Time Normalization
```
1-bit: sign(data)
Running Absolute Mean: data / smooth(|data|)
```
- Reduces influence of large amplitude events
- Emphasizes continuous noise signals
- Improves stability of correlations

### 5. Cross-Correlation
```
For each channel pair (i,j):
    CCF(i,j) = IFFT(FFT(i) * conj(FFT(j)))
```
- Compute for all channel pairs or specific geometries
- GPU-accelerated for speed
- Produces time-lagged correlation functions

### 6. Stacking
```
Stack CCFs over time → Improve signal-to-noise ratio
```
- Linear stacking: simple average
- Phase-weighted stacking: weight by phase coherence
- Enhances coherent signals, suppresses noise

## GPU Acceleration

The correlation workflow uses CUDA.jl for GPU acceleration:

**Check GPU availability**:
```julia
using CUDA
CUDA.functional()  # Should return true
```

**GPU memory management**:
```julia
# Clear GPU memory if needed
CUDA.reclaim()
```

**Processing large datasets**:
- Data automatically transferred to GPU for FFT and correlation
- Results transferred back to CPU for storage
- Batch processing to manage GPU memory

## Parameter Guidelines

### Frequency Bands

| Wave Type | Frequency Range | Typical Use |
|-----------|----------------|-------------|
| Surface waves | 2-8 Hz | Shallow structure, ice properties |
| P-waves | 10-40 Hz | Deep structure, velocity changes |
| S-waves | 5-20 Hz | Shear wave velocity, ice rheology |

### FK Filter Parameters

| Wave Type | cmin (m/s) | cmax (m/s) |
|-----------|-----------|-----------|
| Surface waves | 100 | 500 |
| S-waves | 1500 | 3000 |
| P-waves | 3000 | 6000 |

### Correlation Windows

- `cc_len`: Length of data to correlate
  - Longer → better SNR, worse temporal resolution
  - Shorter → better temporal resolution, worse SNR
  - Typical: 1-2 hours (3600-7200 s)

- `cc_step`: Time between correlations
  - Controls temporal sampling of dv/v
  - Typical: 30-60 minutes (1800-3600 s)

- `maxlag`: Maximum correlation lag
  - Should capture expected wave arrivals
  - Longer → more memory, slower
  - Typical: 5-20 seconds

## Output Files

### Correlation Results (JLD2)
```julia
Dict(
    "corr" => correlation_matrix,  # [lag, chan_i, chan_j, time]
    "time" => time_vector,
    "lag" => lag_vector,
    "channels" => channel_numbers,
    "params" => processing_parameters
)
```

### Autocorrelation Results
- Same format but diagonal of correlation matrix
- Used for dv/v monitoring at single locations
- Faster to compute than full cross-correlations

## Publication Scripts (`Olinger_etal_2026_scripts/`)

These scripts contain finalized analysis for the forthcoming publication.

### `autocorrelation_dvv.ipynb`
- Load autocorrelation results
- Compute dv/v time series using stretching method
- Compare with environmental data (temperature, melt, etc.)
- Generate publication figures

### `surface_cross_correlation.ipynb`
- Surface wave dispersion analysis
- Cross-correlation analysis for structure imaging
- Velocity model estimation

### `plot_study_site.ipynb`
- Create study area maps
- Plot DAS cable geometry
- Show geological/glaciological context

### `ice_flow.ipynb`
- Ice flow modeling
- Stress and strain calculations
- Relate to seismic observations

## Tips and Best Practices

1. **Start Small**: Test on short time windows first, then scale up
2. **Monitor GPU Memory**: Use `CUDA.memory_status()` to check usage
3. **Save Intermediate Results**: Save preprocessed data before correlation
4. **Quality Control**: Always plot raw correlations before stacking
5. **Parameter Sensitivity**: Test different filter parameters systematically
6. **Parallel Processing**: Use Julia's `Distributed` for multi-day processing

## Common Issues

**Issue**: Out of GPU memory
- **Solution**: Reduce number of channels, shorten time windows, or process in batches

**Issue**: Correlations look noisy
- **Solution**: Increase stacking time, adjust whitening/normalization, check data quality

**Issue**: No coherent signals in correlations
- **Solution**: Check FK filter parameters, verify data preprocessing, increase frequency band

**Issue**: Slow processing
- **Solution**: Ensure GPU is being used, reduce maxlag, optimize channel geometry

## Requirements

- Julia 1.8.5
- GPU with CUDA support (optional but recommended)
- See `Project.toml` for package dependencies

Key packages:
- `SeisNoise` - Seismic noise correlation
- `CUDA` - GPU acceleration
- `FFTW` - Fast Fourier transforms
- `JLD2`, `HDF5` - Data storage

## References

- Bensen et al. (2007), "Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements"
- Sens-Schönfelder & Wegler (2006), "Passive image interferometry and seasonal variations of seismic velocities at Merapi Volcano"
- Shapiro & Campillo (2004), "Emergence of broadband Rayleigh waves from correlations of the ambient seismic noise"
- Weaver et al. (2011), "On the precision of noise correlation interferometry"
