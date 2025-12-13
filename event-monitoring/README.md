# Event Monitoring

This folder contains tools and workflows for detecting and characterizing seismic events (icequakes) in DAS data from Store Glacier.

## Contents

- `detection/` - Core detection algorithms and utilities
  - `stalta.py` - STA/LTA (Short-Term Average / Long-Term Average) detection
  - `template_matching.py` - Template-based event detection via cross-correlation
  - `waveforms.py` - Waveform reading, filtering, and plotting utilities
  
- `icequake_search.ipynb` - Main notebook for icequake detection workflow
- `icequake_phase_picking.ipynb` - Machine learning-based phase picking

## Workflows

### 1. Icequake Detection (STA/LTA + Template Matching)

**Notebook**: `icequake_search.ipynb`

This workflow combines traditional STA/LTA detection with template matching for robust event identification.

**Steps**:
1. Load DAS SEGY data for a time window
2. Preprocess data (bandpass filter, decimate if needed)
3. Apply STA/LTA algorithm to identify potential events
4. Extract event waveforms
5. Use template matching to refine detections and identify event families
6. Save detection catalog with event times and metadata

**Key Parameters**:
- `sta_window`: Short-term average window (e.g., 0.5-2 seconds)
- `lta_window`: Long-term average window (e.g., 10-30 seconds)
- `trigger_on`: Threshold for event trigger (e.g., 3-5)
- `trigger_off`: Threshold to end event (e.g., 1-2)
- `frequency_band`: Bandpass filter range (e.g., 2-15 Hz for icequakes)

**Output**:
- Detection times and locations (channel numbers)
- Event waveforms
- Detection statistics (rates, magnitudes if computed)

### 2. Phase Picking with Machine Learning

**Notebook**: `icequake_phase_picking.ipynb`

Automated P-wave and S-wave phase picking using pre-trained machine learning models.

**Methods Used**:
- **SeisBench**: Deep learning models trained on global earthquake catalogs
- **ELEP**: Ensemble Learning for Earthquake Prediction - custom ensemble coherence method

**Steps**:
1. Load detected events from icequake search
2. Prepare waveforms for ML models (resampling, normalization)
3. Run SeisBench models (e.g., PhaseNet, EQTransformer) for phase picks
4. Run ELEP ensemble coherence for refined picks
5. Compare and combine picks from multiple models
6. Generate phase arrival catalog

**Key Parameters**:
- `model_name`: SeisBench model to use (PhaseNet, EQTransformer, etc.)
- `threshold`: Probability threshold for phase detection (0.3-0.7)
- `blinding`: Time window to suppress secondary picks (e.g., 0.5 seconds)

**Output**:
- P-wave and S-wave arrival times
- Pick uncertainties and probabilities
- Phase association results

## Detection Module (`detection/`)

### STA/LTA Detection (`stalta.py`)

Classic earthquake detection algorithm based on the ratio of short-term to long-term signal amplitude.

**Key Functions**:
- `classic_sta_lta()`: Traditional STA/LTA implementation
- `recursive_sta_lta()`: Memory-efficient recursive version
- `trigger_onset()`: Identify trigger on/off times

**Usage**:
```python
from detection.stalta import classic_sta_lta, trigger_onset

# Compute STA/LTA characteristic function
cft = classic_sta_lta(data, nsta=50, nlta=1000)

# Find triggers
triggers = trigger_onset(cft, thr_on=3.5, thr_off=1.0)
```

### Template Matching (`template_matching.py`)

Cross-correlation based detection using known event templates.

**Key Functions**:
- `compute_cc()`: Compute cross-correlation between template and data
- `match_filter()`: Matched filter detection
- `find_event_families()`: Cluster similar events

**Usage**:
```python
from detection.template_matching import match_filter

# Use template to find similar events
detections = match_filter(template, data, threshold=0.7)
```

### Waveform Utilities (`waveforms.py`)

Helper functions for loading and processing DAS waveforms.

**Key Functions**:
- `read_segy()`: Read DAS SEGY files
- `bandpass_filter()`: Apply frequency filter
- `plot_waveforms()`: Visualization tools
- `extract_event()`: Extract event window from continuous data

## Example Workflow

```python
# 1. Import detection tools
from detection.stalta import classic_sta_lta, trigger_onset
from detection.waveforms import read_segy, bandpass_filter

# 2. Load DAS data
data, times, channels = read_segy('DAS_file.sgy')

# 3. Filter data
data_filt = bandpass_filter(data, freqmin=2, freqmax=15, fs=1000)

# 4. Run STA/LTA on each channel
triggers_all = []
for i, channel_data in enumerate(data_filt):
    cft = classic_sta_lta(channel_data, nsta=100, nlta=2000)
    triggers = trigger_onset(cft, thr_on=4.0, thr_off=1.5)
    triggers_all.append(triggers)

# 5. Process and save detections
# ... (see notebooks for full workflow)
```

## Tips and Best Practices

1. **Parameter Tuning**: STA/LTA parameters depend on:
   - Dominant frequency of events (affects window lengths)
   - Noise level (affects thresholds)
   - Event rate (affects how many false positives are acceptable)

2. **Multi-channel Processing**: 
   - Process channels in parallel for speed
   - Stack detections across channels for robust picking
   - Use spatial coherence to reduce false positives

3. **GPU Acceleration**:
   - ML phase picking can use GPU if available
   - Set `device='cuda'` in SeisBench models

4. **Quality Control**:
   - Always manually inspect a subset of detections
   - Compare results from different methods (STA/LTA vs template)
   - Check pick residuals and uncertainties

## Requirements

See main repository `requirements.txt` for dependencies. Key packages:
- `obspy` - Seismological data handling
- `scipy` - Signal processing
- `torch` - PyTorch for ML models
- `seisbench` - Seismology ML models
- Custom `ELEP` package for ensemble picking

## References

- STA/LTA: Withers et al. (1998), "A comparison of select trigger algorithms for automated global seismic phase and event detection"
- Template Matching: Gibbons & Ringdal (2006), "The detection of low magnitude seismic events using array-based waveform correlation"
- PhaseNet: Zhu & Beroza (2019), "PhaseNet: a deep-neural-network-based seismic arrival-time picking method"
