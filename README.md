# EEG JEPA Research

This repository contains experiments for reproducing and analyzing downstream EEG classification results from **S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention**.

The current focus of the repo is:

- loading **Lee2019 SSVEP** through **MOABB**
- applying the downstream preprocessing used in the project
- fine-tuning the **SignalJEPA pre-local** downstream model from **Braindecode**
- reproducing the reported **16s pretraining / 60% masking** SSVEP setting
- diagnosing hard subjects such as **51** and **54**

## Usage

### 1. Create a virtual environment

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Register the environment as a notebook kernel

```bash
python -m ipykernel install --user --name eeg-jepa --display-name "Python (eeg-jepa)"
```

After that, select **Python (eeg-jepa)** as the kernel in Jupyter or VS Code.

### 4. Run the notebooks

Open the project notebooks and run them with the registered kernel.

Typical notebooks/scripts in this repo are used for:

- downstream fine-tuning runs
- experiment sweeps
- artifact evaluation
- subject-level diagnostics

### 5. Verify the environment inside the notebook

Run this in a notebook cell to confirm the kernel is using the expected environment:

```python
import sys
import torch
import mne
import braindecode
import moabb

print("Python executable:", sys.executable)
print("torch:", torch.__version__)
print("mne:", mne.__version__)
print("braindecode:", braindecode.__version__)
print("moabb:", moabb.__version__)
```

### 6. Expected outputs

Successful runs should create artifact folders containing outputs such as:

- `config.json`
- `cv_results.json`
- `subject_metrics.json`
- `global_metrics.json`
- `run_metadata.json`

These artifacts are used to compare runs, inspect subject-level performance, and debug reproduction gaps.

## Notes

- For reproducibility, use the versions pinned in `requirements.txt`.
- If results differ across machines, first verify:
  - package versions
  - selected notebook kernel
  - number of loaded recordings/sessions
  - total retained windows
- The project is sensitive to dataset-loading differences, especially across MOABB versions.

## Current objective

The main reproduction target is the **SSVEP downstream result** associated with:

- **16s pretraining**
- **60% masking**
- **pre-local downstream architecture**
- **5-fold within-subject cross-validation**
