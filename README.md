# EEG JEPA Research

This repository contains experiments for reproducing and analyzing downstream EEG classification results from **S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention**.  # noqa: E999, E701

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
