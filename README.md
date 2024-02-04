# DARSEP
Here we propose DARSEP, a reinforcement learning-based self-game sequence optimization and RetNet learning network-based SARS-CoV-2 spike protein sequence modelling approach that can predict evolutionary sequences and explore antigenic evolution.
## Components
- `data` : Required data
- `src` : Python code of DARSEP-SPRLM and DARSEP-PMLM
- `results` : Directory for the results
- `analysis` : Jupyter notebooks for all the analysis and plots
## Requirements
### Hardware requirements
- Trained and tested on one NVIDIA RTX 3090 with 24GB GPU memory.
- For storing all intermediate files for all methods and all datasets, approximately 8G of disk space will be needed.
### Software requirements
In conda environment, follow the following command to install the necessary environment:
```
pip install -r requirements.txt
```
## Data downlaod
GISAD dataset repuires authentication, and registration is needed to access the data. Therefore, we can't provide the data directly. So we download the data from their web: https://www.gisaid.org.And we pre-process the data to match the model inputs.
