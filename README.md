# greenlanDAS

**DAS (Distributed Acoustic Sensing) Analysis for Store Glacier, Greenland**

This repository contains analysis code for processing and interpreting DAS data from Store Glacier in Greenland. The project combines Python and Julia for various seismological tasks including event detection, ambient noise correlation, and glacier structure monitoring.

## Overview

This codebase supports three main analysis categories:

- **Event Monitoring**: Detection and phase picking of icequakes using STA/LTA algorithms, template matching, and machine learning models
- **Data Exploration**: Spectrogram computation, noise characterization, and data quality assessment
- **Structure Monitoring**: Ambient noise correlation, velocity change (dv/v) analysis, and glacier structure imaging

## Repository Structure

```
greenlanDAS/
├── event-monitoring/          # Icequake detection and phase picking
│   ├── detection/            # Detection algorithms (STA/LTA, template matching)
│   ├── icequake_search.ipynb
│   └── icequake_phase_picking.ipynb
├── data-exploration/          # Data quality and characterization
│   ├── spectrograms.ipynb
│   ├── compute_spectrogram.py
│   ├── characterize_noise.ipynb
│   ├── run_rms.jl
│   └── file_wrangling.ipynb
├── structure-monitoring/      # Correlation analysis
│   ├── correlation/          # Main correlation workflow (Julia)
│   └── Olinger_etal_2026_scripts/  # Publication analysis scripts
├── deprecated/                # Archived older versions
├── issues/                    # Troubleshooting and debugging
├── obspy_local/              # Custom ObsPy modifications (if needed)
├── requirements.txt          # Python dependencies
├── Project.toml              # Julia dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- **Python**: 3.8, 3.10, or 3.11
- **Julia**: 1.8.5 or compatible version
- **GPU** (optional but recommended): CUDA-capable GPU for accelerated correlation and detection
- **CUDA Toolkit**: 11.8 or 12.1 (if using GPU)

### Python Environment Setup

#### Option 1: Using pip

```bash
# Create a virtual environment
python -m venv greenlandas_env
source greenlandas_env/bin/activate  # On Windows: greenlandas_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install ELEP (custom ensemble learning package)
git clone https://github.com/Denolle-Lab/ELEP.git
cd ELEP
pip install -e .
cd ..
```

#### Option 2: Using conda

```bash
# Create conda environment
conda create -n greenlandas python=3.11
conda activate greenlandas

# Install dependencies
pip install -r requirements.txt

# Install ELEP
git clone https://github.com/Denolle-Lab/ELEP.git
cd ELEP
pip install -e .
cd ..
```

#### GPU Support for PyTorch

If you have a CUDA-capable GPU, install PyTorch with CUDA support:

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### Julia Environment Setup

1. **Install Julia 1.8.5** from [julialang.org](https://julialang.org/downloads/)

2. **Activate the project environment**:

```bash
cd /path/to/greenlanDAS
julia --project=.
```

3. **Install dependencies** (in Julia REPL):

```julia
using Pkg
Pkg.instantiate()
```

This will install all packages specified in `Project.toml` with compatible versions.

4. **Set up GPU support** (optional, for CUDA acceleration):

```julia
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.functional()  # Should return true if GPU is available
```

## Usage

### Event Monitoring

Navigate to `event-monitoring/` for icequake detection and phase picking workflows.

**Quick start: Detect icequakes**

```bash
cd event-monitoring
jupyter notebook icequake_search.ipynb
```

This notebook demonstrates:
- Loading DAS SEGY data
- Applying STA/LTA detection algorithm
- Template matching for event identification
- Saving detection catalogs

**Phase picking with machine learning**

```bash
jupyter notebook icequake_phase_picking.ipynb
```

Uses seisbench and ELEP ensemble models for automated P- and S-wave phase picking.

See `event-monitoring/README.md` for detailed workflow descriptions.

### Data Exploration

Navigate to `data-exploration/` for data quality assessment and characterization.

**Compute spectrograms**

```python
python compute_spectrogram.py
# Or use the interactive notebook:
jupyter notebook spectrograms.ipynb
```

**Characterize noise levels**

```bash
# Python notebook for noise analysis
jupyter notebook characterize_noise.ipynb

# Or Julia script for RMS computation
julia run_rms.jl
```

See `data-exploration/README.md` for detailed usage instructions.

### Structure Monitoring

Navigate to `structure-monitoring/` for ambient noise correlation and velocity change analysis.

**Run correlation workflow** (Julia, GPU-accelerated):

```bash
cd structure-monitoring/correlation
julia --project=../.. run_correlation.jl
```

**Batch processing for different wave types**:

```bash
cd structure-monitoring/correlation/batch
julia --project=../../.. run_correlation_p.jl    # P-wave correlations
julia --project=../../.. run_correlation_s.jl    # S-wave correlations
julia --project=../../.. run_surface_correlation.jl  # Surface wave correlations
```

**Post-processing and visualization**:

```bash
cd structure-monitoring/correlation
jupyter notebook postprocessing.ipynb
jupyter notebook autocorrelation_visualization.ipynb
```

See `structure-monitoring/README.md` for detailed correlation workflows and parameter descriptions.

### Publication Scripts

The `structure-monitoring/Olinger_etal_2026_scripts/` folder contains finalized analysis code for the forthcoming publication:

- `autocorrelation_dvv.ipynb`: Velocity change analysis from autocorrelations
- `surface_cross_correlation.ipynb`: Surface wave cross-correlation analysis
- `plot_study_site.ipynb`: Study area maps and figures
- `ice_flow.ipynb`: Ice flow stress/strain modeling

## Data Format

This codebase primarily works with:
- **Input**: SEGY format DAS data (1 kHz and 4 kHz sampling rates)
- **Output**: HDF5 (.h5) and JLD2 (.jld2) for processed results
- **Metadata**: Station XML files for instrument response

## GPU Acceleration

Many workflows support GPU acceleration for faster processing:

- **Python**: PyTorch-based detection and phase picking
- **Julia**: CUDA.jl-accelerated correlation computations

GPU usage is automatic if a CUDA-capable device is detected. CPU fallback is available.

## Common Workflows

### 1. Event Detection Pipeline
```
Load SEGY → Filter/Preprocess → STA/LTA Detection → Template Matching → Phase Picking → Catalog
```

### 2. Correlation Workflow
```
Load SEGY → Resample → Detrend/Taper → Bandpass → FK Filter → Whitening → 
1-bit Normalization → Cross-Correlate → Stack → Save JLD2
```

### 3. Velocity Change Analysis
```
Load Autocorrelations → Time Stacking → Frequency Filtering → 
Stretching Method → Compute dv/v → Visualization
```

## Troubleshooting

- **SEGY reading issues**: Check `issues/` folder for known problems and solutions
- **GPU errors**: Verify CUDA installation with `nvidia-smi` (Linux/Windows) or system settings (macOS)
- **Julia package conflicts**: Try `Pkg.resolve()` or `Pkg.update()` in Julia REPL
- **Python import errors**: Ensure all packages from `requirements.txt` are installed

## Contributing

This is research code for the Denolle Lab. For questions or contributions, please contact:
- Stephanie Olinger
- Marine Denolle (mdenolle@uw.edu)

## Citation

If you use this code, please cite:

```
Olinger, S. D., et al. (2026). [Title]. [Journal]. [In preparation]
```

## License

MIT License - See `LICENSE` file for details.

## Acknowledgments

This work was supported by [funding sources]. DAS data collection was conducted at Store Glacier, Greenland.
