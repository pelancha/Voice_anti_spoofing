# LCNN model to detect voice spoofing

This script allows to detect voice spoofing based on [ASVspoof 2019 Dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset), which includes spoofing generated via different technologies (TTS and VC algorithms) and replay scenarios. Full information on the dataset can be viewed at [ASVspoof 2019 Evaluation Plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf).

Current implementation of the script allows to achieve EER of 2.7% and supports use only in Kaggle environment.

## Implemented features
- LCCN architechture is based on [the paper of the Speech Technology Center](https://arxiv.org/abs/1904.05576)
- Dropout layer is implemented as in [the paper by Xinyue Ma et al.](https://ieeexplore.ieee.org/document/9428313)
- Frontend is taken from [A Comparative Study on Recent Neural Spoofing Countermeasures](https://arxiv.org/abs/2103.11326)

**More information on the frontend:**
- No data augmentation is used
- Magnitude spectrogram, compressed to 60-dimensional hidden features, was used as model input
- LCNN-trim-pad combination was used; the input was padded with cyclicly repeating parts of the initial audio 

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
