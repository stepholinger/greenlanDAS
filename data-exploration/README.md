# Data Exploration

This folder contains tools for exploring DAS data quality, computing spectrograms, characterizing noise, and performing data preprocessing.

## Contents

- `spectrograms.ipynb` - Interactive spectrogram computation and visualization
- `compute_spectrogram.py` - Python script for batch spectrogram generation
- `characterize_noise.ipynb` - Noise level analysis and characterization
- `run_rms.jl` - Julia script for computing RMS (Root Mean Square) values
- `file_wrangling.ipynb` - Data preprocessing, format conversion, and resampling

## Workflows

### 1. Spectrogram Analysis

**Files**: `spectrograms.ipynb`, `compute_spectrogram.py`

Spectrograms provide a time-frequency representation of DAS data, useful for:
- Identifying dominant frequency content
- Detecting transient signals and events
- Understanding noise characteristics over time
- Quality control and data validation

**Key Features**:
- Compute spectrograms using Short-Time Fourier Transform (STFT)
- Visualize multiple channels simultaneously
- Customize frequency range and time resolution
- Export high-resolution figures

**Usage (Notebook)**:
```bash
jupyter notebook spectrograms.ipynb
```

**Usage (Script)**:
```python
python compute_spectrogram.py --input DAS_file.sgy --output spectrograms/ --fmin 0.1 --fmax 50
```

**Key Parameters**:
- `nperseg`: Window length for FFT (controls frequency resolution)
- `noverlap`: Overlap between windows (controls time resolution)
- `fmin`, `fmax`: Frequency range to display
- `vmin`, `vmax`: Color scale limits (in dB)

**Output**:
- Spectrogram images (PNG)
- Time-frequency arrays (HDF5)
- Summary statistics

### 2. Noise Characterization

**File**: `characterize_noise.ipynb`

Analyze noise levels and characteristics across the DAS array.

**Analysis Types**:
- **Temporal noise variation**: How noise changes over time
- **Spatial noise patterns**: Noise levels across different channels
- **Frequency content**: Dominant noise frequencies
- **Power Spectral Density (PSD)**: Noise power vs frequency
- **Probability Density Functions (PDF)**: Amplitude distributions

**Steps**:
1. Load DAS data segments
2. Compute RMS, variance, or PSD for each channel
3. Identify noisy channels or time periods
4. Compare to reference models (e.g., New Low/High Noise Models)
5. Generate noise summary statistics

**Output**:
- Noise level time series
- Spatial noise maps
- PSD plots
- Channel quality metrics

### 3. RMS Computation

**File**: `run_rms.jl`

Julia script for efficient computation of running RMS values on DAS data.

**Purpose**:
- Fast RMS computation for large datasets
- Sliding window RMS for temporal variability
- Multi-channel parallel processing
- Memory-efficient for long time series

**Usage**:
```bash
julia --project=.. run_rms.jl
```

**Key Parameters** (edit in script):
- `window_length`: RMS window duration (e.g., 10 seconds)
- `overlap`: Overlap between windows
- `channels`: Channel range to process
- `output_file`: Where to save results (HDF5 or JLD2)

**Output**:
- RMS time series for each channel
- Summary statistics (mean, median, percentiles)
- Quality flags for noisy periods

### 4. File Wrangling and Preprocessing

**File**: `file_wrangling.ipynb`

Data preprocessing and format conversion utilities.

**Capabilities**:
- **Format conversion**: SEGY → HDF5, downsampled versions
- **Resampling**: Change sampling rate (e.g., 4000 Hz → 1000 Hz)
- **Segmentation**: Split continuous data into daily/hourly files
- **Channel selection**: Extract subsets of channels
- **Quality checks**: Verify data integrity, check for gaps

**Common Tasks**:

1. **Downsample data**:
```python
from scipy.signal import decimate

# Decimate by factor of 4: 4000 Hz → 1000 Hz
data_ds = decimate(data, q=4, axis=0)
```

2. **Convert SEGY to HDF5**:
```python
import h5py

# Save preprocessed data
with h5py.File('DAS_preprocessed.h5', 'w') as f:
    f.create_dataset('data', data=data)
    f.create_dataset('time', data=time)
    f.create_dataset('channels', data=channels)
```

3. **Extract time window**:
```python
# Extract 1-hour segment
start_idx = int(start_time * fs)
end_idx = int(end_time * fs)
data_segment = data[start_idx:end_idx, :]
```

**Output**:
- Preprocessed HDF5 files
- Resampled data
- Metadata files

## Example Workflows

### Quick Data Quality Check

```python
# 1. Load a segment of data
from obspy_local.segy import _read_segy
st = _read_segy('DAS_file.sgy')

# 2. Compute and plot spectrogram
from scipy.signal import spectrogram
f, t, Sxx = spectrogram(st[0].data, fs=1000, nperseg=1024)

import matplotlib.pyplot as plt
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='PSD (dB)')
plt.show()

# 3. Check RMS levels
rms = np.sqrt(np.mean(st[0].data**2))
print(f"RMS: {rms:.2e}")
```

### Batch Spectrogram Processing

```bash
# Process all SEGY files in a directory
for file in *.sgy; do
    python compute_spectrogram.py --input $file --output spectrograms/
done
```

### Identify Noisy Channels

```python
# Compute RMS for all channels
rms_all = np.sqrt(np.mean(data**2, axis=0))

# Find channels with RMS > 3x median
median_rms = np.median(rms_all)
noisy_channels = np.where(rms_all > 3 * median_rms)[0]

print(f"Noisy channels: {noisy_channels}")
```

## Tips and Best Practices

1. **Spectrogram Resolution**:
   - Larger `nperseg` → better frequency resolution, worse time resolution
   - Smaller `nperseg` → better time resolution, worse frequency resolution
   - Typical values: 256-2048 samples depending on fs and application

2. **Noise Analysis**:
   - Always compare to quiet reference periods
   - Consider diurnal and seasonal variations
   - Check for cultural noise sources (regular patterns)

3. **Data Quality**:
   - Plot raw data before processing to check for glitches
   - Verify sampling rate and channel count
   - Check for time gaps or jumps

4. **Performance**:
   - Use Julia (`run_rms.jl`) for very large datasets
   - Process channels in parallel when possible
   - Downsample first if high frequency content not needed

5. **File Formats**:
   - HDF5 for large numerical arrays (faster I/O)
   - SEGY for raw data (standard seismic format)
   - JLD2 for Julia-specific data structures

## Common Issues

**Issue**: Spectrograms show discontinuities or artifacts
- **Solution**: Check for data gaps, verify detrending, adjust window parameters

**Issue**: Memory errors when loading large SEGY files
- **Solution**: Process in chunks, use memory-mapped arrays, or downsample first

**Issue**: Noise levels vary dramatically across channels
- **Solution**: This is normal for DAS! Check cable coupling, burial depth, and environmental factors

## Requirements

See main repository `requirements.txt` for Python dependencies. Key packages:
- `numpy`, `scipy` - Numerical computing and signal processing
- `matplotlib` - Visualization
- `h5py` - HDF5 file I/O
- `obspy` - Seismological data formats

For Julia scripts:
- Julia 1.8.5
- Packages: `HDF5`, `JLD2`, `Statistics`, `FFTW`

## References

- Welch's method for PSD: Welch (1967), "The use of fast Fourier transform for the estimation of power spectra"
- DAS noise characteristics: Lindsey et al. (2020), "Fiber-optic network observations of earthquake wavefields"
- Spectrogram analysis: Oppenheim & Schafer (2009), "Discrete-Time Signal Processing"
