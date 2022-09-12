# DGNN_analysis
This repository contains the code for Dynamic Graph Neural Networks on Hardware: Bottleneck Analysis (IISWC 2022).

## NVIDIA Nsight Systems Installation

Please install the profiling tool using this link: https://developer.nvidia.com/nsight-systems

Directly run the downloaded installer to install this tool on your machine.

## Pytorch Profiler Installation

#### 1. Install PyTorch and Torchvision using the following command:

```{bash}
pip install torch torchvision
```
#### 2. Install PyTorch Profiler TensorBoard Plugin for visualization:

```{bash}
pip install torch_tb_profiler
```

#### 3. Open the generated trace file using this command:

```{bash}
tensorboard --logdir=./log
```
log is a folder containing trace files. You can set it to any path containing the generated trace files.

## Profiling results:

Once you profile the models using Pytorch Profiler, you will get results as follows:

![image](https://github.com/eun4231/DGNN_analysis/blob/main/Pytorch_profiler.png)

If you choose to use NVIDIA Nsight Systems, you will get similiar results below:

![image](https://github.com/eun4231/DGNN_analysis/blob/main/NS.png)
