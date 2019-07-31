# README #

This is the code for the following paper:

Kathan Vyas, Rui Ma, Behnaz Rezaei, Shuangjun Liu, Thomas Ploetz, Ronald Oberleitner and Sarah Ostadabbas, “Recognition of Atypical Behavior in Autism Diagnosis from Video using Pose Estimation Over Time,” IEEE International Workshop on Machine Learning for Signal Processing (MLSP’19), October 13-16, 2019, Pittsburgh, PA, USA. 


Contact: 
[Kathan Vyas](vyas.k@husky.neu.edu)

[Sarah Ostadabbas](ostadabbas@ece.neu.edu)

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
This will install Caffe2. Once Caffe2 is installed,  next thing to be done is to install [Detect and Track Code] (https://github.com/facebookresearch/DetectAndTrack). Please follow the instruction as informed there to install the algorithm code. 
Next please replace the tools/test_on_single_video.py from our directory to the Detect and Track Tools/test_on_single_video.py. copy all the otehr files from our directory to Detec and Track home directory.

You can run Pose_Estimation.py.

### 4. PoTion ###

- Install [Tensorflow] (https://www.tensorflow.org/install)
There are three different types of PoTios one can use. 

1. PoTion: If the pose-estimation results are good and all required keypoints are obtained, one can run PoTion.py to run the normal poTion program.
2. PoTion + Linear Interpolation: If the results have 10% keypoints missing, one can use poTion_Linear.py file to implement linear interpolation and then apply PoTion algorithm.
3. PoTion + Particle Filter: With more than 10% missing keypoints, one can use poTion with particle filter using the file potion_pf.py.

### 5. Classification ###

Once The results of Potion are obtained, the PoTion images could be used to run potion/potion_data/Classification.py


## Citation 
If you found our work useful in your research, please consider citing our paper:

```
@article{vya2019recognition,
  title={Recognition of Atypical Behavior in Autism Diagnosis from Video using Pose Estimation Over Time},
  author={Kathan Vyas, Rui Ma, Behnaz Rezaei, Shuangjun Liu, Thomas Ploetz, Ronald Oberleitner and Sarah Ostadabbas},
  journal={IEEE International Workshop on Machine Learning for Signal Processing (MLSP’19), October 13-16, 2019, Pittsburgh, PA, USA.},
  year={2019}
}
```


## License 
* This code is for non-commertial purpose only. For other uses please contact ACLab of NEU. 
* No maintainence survice 

### Acknowledgements ###

The code was built upon an initial version of the [Detect and Track](https://github.com/facebookresearch/DetectAndTrack) code base. Many thanks to the original authors for making their code available!



