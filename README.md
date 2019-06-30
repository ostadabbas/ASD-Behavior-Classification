# README #

This is the code for the following paper:
"Recognition of Atypical Behavior in Autism Diagnosis from Videos using Pose Estimation over Time"

Check the Project on the page:
PAGE URL

Contact: 
[Kathan Vyas](vyas.k@husky.neu.edu)
[Rui MA](ma.rui@husky.neu.edu)

### Contents ###

* [1. Requirements](#1-Requirements)
* [2. Setting up Dataset](#2-setting_up_Dataset)
* [3. Using Pose Estimation](#3-Pose_estimation)
* [4. Using PoTion Representation Algorithm](#4-PoTion)
* [5. Classification](#5-Classification)
* [Citation](#Citation)
* [Lisence](#Lisence)
* [Acknowledgements](#Acknowledgements)


### 1. Requirements ###

Due to dependency issues between different frameworks, there are two different enviornemnts which are needed to be setup for execution of different processes. 
We work around two frameworks - Caffe and Tensorflow. This is primarily because of the algorithms in use. Caffe is needed to run the Pose Estimation network which is a version of Detect and Track based on Facebook's Detectron.
For rest of all algorithms including PoTion and Classification, we have used Tensorflow. There are two requirements files that solve enviornments for both the frameworks. 

The requirements to run all the algorithms provided in this paper include python 2 and python 3. 

Please follow the steps in order to prepare the enviornments.



* Preparing for Pose Estimation

Install [python 2.7] (https://www.python.org/downloads/)
For pose estimation the requirement file is requirement_caffe.txt. 
Please create a conda enviornment using this file which contains all the necessary libraries and dependencies. Once you create the enviornment, you can follow the instructions provided in pose estimation section.

* Preparing for PoTion and Classification

Install [python 3.5 or higher] (https://www.python.org/downloads/)
For pose estimation the requirement file is requirement_tensor.txt. 
Please create a conda enviornment using this file which contains all the necessary libraries and dependencies. Once you create the enviornment, you can follow the instructions provided in poTion and Classification section.

### 2. Setting up Dataset ###

Your tagged video dataset should be kept in a folder separately and should make changes in the Pose_Estimation.py. Please change the parse_arguments in the file to your dataset directory.

### 3. Pose Estimation ###

- Install [Caffe2] (https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile) or follow these instructions:

```
$ cd ..
$ git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
$ git submodule update --init
$ mkdir build && cd build
$ export CONDA_PATH=/path/to/anaconda2  # Set this path as per your anaconda installation
$ export CONDA_ENV_PATH=$CONDA_PATH/envs/$ENV_NAME
$ cmake \
	-DCMAKE_PREFIX_PATH=$CONDA_ENV_PATH \
	-DCMAKE_INSTALL_PREFIX=$CONDA_ENV_PATH \
	-Dpybind11_INCLUDE_DIR=$CONDA_ENV_PATH/include \
	-DCMAKE_THREAD_LIBS_INIT=$CONDA_ENV_PATH/lib ..
$ make -j32
$ make install -j32  # This installs into the environmen
```
This will install Caffe2. Once Caffe2 is installed, you can run Pose_Estimation.py.

### 4. PoTion ###

- Install [Tensorflow] (https://www.tensorflow.org/install)
There are three different types of PoTios one can use. 



### 5. Classification ###

### Citation ###

### Lisence ###

### Acknowledgements ###



