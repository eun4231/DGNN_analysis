# DGNN_analysis
This repository contains the code for Dynamic Graph Neural Networks on Hardware: Bottleneck Analysis (IISWC 2022).

## NVIDIA Nsight Systems download

Please download this profiling tool usig this link: https://developer.nvidia.com/nsight-systems

Directly run the donloaded installer to install this tool on your machine.

## Pytorch Profiler Installation

#### 1. Install Pytorch and Torchvision using the following command:

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
Log is a folder containing trace files. Or you can change it to any path containing generated trace files.
