# SDSC4016 Fundamentals of Machine Learning II

SDSC4016 Homework 4:

## Description

Speaker Classification Problem

## Getting Started

### Dependencies

- Python
  - Python 3.10+
  - Jupyter
  - pytorch

### Install mini-conda and mamba

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda install mamba -n base -c conda-forge
```

### Set up conda environment

```bash
mamba create -n 4016hw4
mamba activate 4016hw4
```

### Installing dependencies

```bash
# conda or mamba
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install -c conda-forge Jupyter ipykernel
mamba install -c conda-forge pandas numpy scikit-learn tqdm
pip install conformer
```

#### Weak Baseline

- [Jupyter Notebook](https://github.com/CityU-SDSC4016-2022/SDSC4016-hw4/blob/notebook/src/Exam%20baseline.ipynb)

#### Strong Baseline

1. You can run it by ```script.sh```

2. Or you can run it by the following command:

    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    source ~/miniconda3/etc/profile.d/mamba.sh
    mamba activate 4016hw4
    python src/script.py
    ```

### Dataset

- Path: ```./data/Dataset/```

### Final Score

- Public: 0.94741
- Private: 0.94833
