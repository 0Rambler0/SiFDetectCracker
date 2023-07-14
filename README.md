# SiFDetectCracker

By Hai Xuan, Liu Xin, 2023

------

This repository contains our implementation of a black-box adversarial attack system against fake voice detection system named SiFDetectCracker which is proposed by us.

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements:
```
$ git clone https://github.com/asvspoof-challenge/2021.git
$ cd SiFDetectCracker
$ conda create --name SiFDetectCracker python=3.6.10
$ conda activate SiFDetectCracker
$ conda install pytorch=1.4.0 -c pytorch
$ pip install -r requirements.txt
```
Deep4SNet use matlab to convert audio to image so you should install matlab if you want to attack Deep4SNet.

## Experiments

### Dataset
ASVspoof 2019 dataset is used in our experiment, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336). We select 195 samples as the test samples by using the utils in dataset utils.

### Create evaluation set

To get the same test samples we used in experiment, please change the path in codes and run:

```
python3 dataset_utils/create_long_audio_protocol.py
python3 dataset_utils/get_long_audio.py
python3 dataset_utils/create_exp_set.py
```

### Evaluation
Normal evaluation:
```
python3 attack_exp.py --target=your_target --lr=0.01 --sigma=0.0001 
```
For Deep4SNet, please set the parameter `sigma=0.1`

No time/noise evaluation
```
python3 attack_exp.py --target=your_target --lr=0.1 --sigma=0.0001 --mode=no_time 

python3 attack_exp.py --target=your_target --lr=0.1 --sigma=0.0001 --mode=no_noise --iteration_num=8
```


