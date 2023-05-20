# Simple Kinesthetic Haptics for Object Recognition

## Introduction

The code is the implementation of the classifiers for object recognition presented in the paper (see *bibtex* at the bottom):

> Avishai Sintov and Inbar Meir, *Simple Kinesthetic Haptics for Object Recognition*, International Journal of Robotics Research, Accepted May 2023.

Video of experiments can be viewed in [YouTube](https://youtu.be/wAWAPW6IfTE)


## Prerequisites

The following Python packages are required in order to run the code:
- Python 3.*
- TensorFlow 2.0
- Pickle
- tqdm
- sklearn

## Generate data

In order to generate grasp samples for a set of objects, do the following:

1. Add **.stl** object to the *cad* sub-folder.
2. Choose the number of fingers in file **gen_data.py**: 
    ```sh
    num_fingers = 4
    ```
3. Choose if the grasps include the normal ('_withN') or not ('_noN') in file **gen_data.py**:
    ```sh
    with_normals = '_withN'
    ```
4. Choose the number of grasp samples per object in file **gen_data.py**:
    ```sh
    M = 10000
    ```
5. Run the file **gen_data.py**.

## Iterative Classification (IC)

First, a Neural-Network classifier must be trained. An example for a 4-finger grasp with normals is included. 
A model can trained using file **train_nn_model.py** where the following lines determine the size of the network, number of fingers to use and whether to use normals:
```sh
H = [307] * 8
num_fingers = 4
with_normals = '_withN'
```
The corresponding dataset is automatically loaded once these are determined.
An example of how to run IC is given through the generation of a confusion matrix in file **IC_inference.py**.

## Bayesian Classification (BC)

File **kde_pdf.py** is a class that generates a Bayesian classifier based on a Kernel Density Estimator. An example of how to run BC is given through the generation of a confusion matrix in file **BC_inference.py**. Here also, one must determine the number of fingers to use and whether to use normals:
```sh
num_fingers = 4
with_normals = '_withN'
```

## Citation
```
@article{Sintov2023,
  Title                    = {Object Recognition using Simple Kinesthetic Haptics},
  Author                   = {Avishai Sintov and Inbar Meir},
  Journal                  = {International Journal of Robotics Research},
  year = {Accepted, May 2023},
}
```


