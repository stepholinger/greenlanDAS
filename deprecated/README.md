# Deprecated Files

This folder contains archived versions of notebooks and scripts that have been superseded by newer implementations. These files are kept for reference and historical purposes.

## Contents

### From Original `deprecated/` Folder

- **`indices.ipynb`** - Early exploration of data indices and metadata structures
  - *Status*: Superseded by `file_wrangling.ipynb` in `data-exploration/`
  - *Date*: [Original development]

- **`read_and_correlate_DAS_backup_2_27.ipynb`** - Backup version of correlation workflow
  - *Status*: Replaced by Julia-based correlation in `structure-monitoring/correlation/`
  - *Date*: February 27 backup
  - *Reason*: Migrated to more efficient Julia implementation with GPU support

- **`read_and_correlate_DAS_SGY_2_9.ipynb`** - Early SEGY reading and correlation
  - *Status*: Replaced by modular workflow
  - *Date*: February 9 version
  - *Reason*: Separated into data exploration and correlation modules

- **`seisnoise_testing.ipynb`** - Testing of SeisNoise.jl package capabilities
  - *Status*: Concepts integrated into production correlation code
  - *Date*: [Package testing phase]
  - *Reason*: Package testing complete, functionality incorporated

### Moved from Root Directory

- **`icequake_search_old.ipynb`** - Previous version of icequake detection workflow
  - *Status*: Superseded by `icequake_search.ipynb` in `event-monitoring/`
  - *Reason*: Improved detection algorithms and reorganized workflow

- **`debugging.ipynb`** - Low-level debugging for SEGY file reading
  - *Status*: Issues resolved, ObsPy functionality stabilized
  - *Reason*: SEGY reading issues fixed, no longer needed for routine analysis

## When to Use These Files

These deprecated files may still be useful for:

1. **Historical Reference**: Understanding how methods evolved
2. **Alternative Approaches**: Comparing different implementation strategies
3. **Troubleshooting**: Identifying where changes were made
4. **Code Recovery**: Retrieving specific functions or approaches that were removed

## Migration Guide

If you need functionality from these deprecated files:

### For Correlation Workflows
**Old**: `read_and_correlate_DAS_backup_2_27.ipynb`, `read_and_correlate_DAS_SGY_2_9.ipynb`

**New**: 
- Use `structure-monitoring/correlation/run_correlation.jl` for correlations
- Use `data-exploration/file_wrangling.ipynb` for SEGY reading/preprocessing
- See `structure-monitoring/README.md` for workflow documentation

**Benefits of New Approach**:
- GPU acceleration with Julia CUDA
- Modular, reusable functions
- Better memory management
- Faster processing for large datasets

### For Event Detection
**Old**: `icequake_search_old.ipynb`

**New**: `event-monitoring/icequake_search.ipynb`

**Improvements**:
- Integration with SeisBench ML models
- Template matching capabilities
- Better documentation and parameter settings
- Modular detection functions in `event-monitoring/detection/`

### For Data Exploration
**Old**: `indices.ipynb`

**New**: `data-exploration/file_wrangling.ipynb`

**Improvements**:
- More comprehensive preprocessing tools
- Format conversion utilities
- Better handling of metadata and time stamps

### For Package Testing
**Old**: `seisnoise_testing.ipynb`

**New**: Functionality is now in `structure-monitoring/correlation/functions/`

**What Changed**:
- SeisNoise.jl functions adapted for DAS-specific needs
- Custom data types defined in `Types.jl`
- Workflow optimized in `Workflow.jl`

## Important Notes

⚠️ **Do Not Modify**: These files are archived as-is for reference only

⚠️ **Not Maintained**: Deprecated files may not work with current package versions

⚠️ **Use Current Versions**: Always use the updated workflows in the main folders for production analysis

## Cleanup Policy

Files in this folder are retained indefinitely for historical reference unless:
- They contain no unique functionality
- They are completely superseded by newer implementations
- They reference data or methods no longer in use

## Questions?

If you're unsure whether to use a deprecated file or the current implementation:
1. Check the main README.md for current workflows
2. Review folder-specific READMEs (`event-monitoring/`, `data-exploration/`, `structure-monitoring/`)
3. Contact the repository maintainers

## File Organization

Files are organized chronologically by when they were deprecated:
- Latest deprecations at the top of this README
- Original `deprecated/` folder contents listed separately
- Date stamps included when known

---

*Last Updated*: December 2025
*Maintained by*: Denolle Lab
