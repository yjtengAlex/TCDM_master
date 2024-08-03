# TCDM_master
A conditional diffusion model
# Installation

1. Create a virtual environment in Anaconda

```shell
conda create -n TCDM python=3.8
```

2. Run the installation command

```shell
pip install -r requirements.txt
conda install -c mrtrix3 mrtrix3
```

# Data preprocessing

The data were denoised, decapitated, normalized and downsampled:

```
python denoiseData.py
python createLow.py
```

We have prepared a script to create the track density imaging:

```
bash create_tck_.sh
bash create_track_density.sh
```

The data conversion process is shown in the figure:

![image](https://github.com/yjtengAlex/TCDM_master/assets/121848586/f19a29aa-9df8-47c0-8f3d-b79a69f862a7)


# Create data set

```
python prepare.py
```



# Training model

```
python MainCondition.py
```



## Generative Model

```
python TestCondition.py
```



# Calculation performance index

```
python cal_FA.py
```



# Show the super resolution effect

![image](https://github.com/user-attachments/assets/a5ac53e0-3054-4c17-9a09-e921690507cb)

