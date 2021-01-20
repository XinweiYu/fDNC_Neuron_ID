# DLC: Deep Learning Correspondence for Neuron ID

This tool was developed for automatically predicting neuron correspondence for point clouds of neurons. The point clouds do not requirestraightening or alignment as preprocessing. 

# Installation
The code was run in python 3.7 environment. The packages and version can be found in requirements.txt

# Usage
The code can be run with or without a Nvidia gpu. Set the argument cuda=False force the model to be run under cpu. The running speed with a Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz is 0.05s/worm.

A template worm with labels already assigned is required. The data should be a python dictionary stored using pickle package with following keys:
1. pts: a numpy array of dimension N*3 as the xyz position of neurons(unit:micrometer)
2. fluo: a numpy array of dimension N*num_channel as the fluorescent signals.(Set to be None if not available)
3. name: a list of neuron names of length N as the name assigned to each neuron.

We predict the neuron names for a test worm, which is also a python dictionary stored with pickle. The dictionary contains 2 keys:
1. pts: a numpy array of dimension M*3 as the xyz position of neurons(unit:micrometer)
2. fluo: a numpy array of dimension M*num_channel as the fluorescent signals.(Set to be None if not available)

The worm should be lie on its right side. (If the worm lies on the left side, we can rotate the worm for 180 degree, which invert the sign of x and z coordinates)

An example of Data of template and test worm is provided in Data/Example

To run the ID prediction, use the predict function in src/DLC_predict.py. An example of running code was provided in src/example.ipynb 

# Pretrain model
We provide a pretrained model and recommend our user to download the pretrained model and save it in model folder as model/model.bin. The model can be found <a href="https://osf.io/t7dzu/">Data And Pretrained Model</a>

We also provide the code for training a new model with synthetic data. An example for training with your own train and validation set is as follows:
```
python ./src/fDLC_train.py --train_path ./Data/train --eval_path ./Data/val
```
We also provide the training data <a href="https://osf.io/t7dzu/">Data And Pretrained Model</a>

