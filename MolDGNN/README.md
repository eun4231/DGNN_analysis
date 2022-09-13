## MolDGNN: Molecular Dynamic Graph Neural Network
This page contains the code for the model MolDGNN and for performing analysis using two profiling tools: Pytorch Profiler and Nsight Systems.

---
## Related Paper and Github Project
- [Geometric learning of the conformational dynamics of molecules using dynamic graph neural networks](https://arxiv.org/abs/2106.13277) [[GitHub](https://github.com/pnnl/mol_dgnn)]

## Dataset
- [Molecular Trajectories rom the ISO17 Dataset](http://quantum-machine.org/datasets)


## Requirements
- Dependencies (with python >= 3.7):
```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

---

## Evaluate MolDGNN Inference

#### Preprocess the data
```
    $ python preprocess.py
```

#### General flags


This code can be given the following command-line arguments:
```{txt}
    1. `--data`: path to the processed data file for test set
    2. `--save_dir`: path to save
    3. `--n_atoms`: number of atoms in system
    4. `--window_size`: window size
    5. `--batch_size`: batch size
    6. `--model`: path to pretrained model
```

#### MolDGNN Inference

To run the inference of MolDGNN model, use the following command:
```
    $ python test.py
```



#### Profiling the Interaction via Nsight Systems
The following commands are example of runnning Nsight Systems to generate profile results for different model configurations:
```
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o prof/nsys/Mol_dGnn_B4 --force-overwrite true python Nsight_Profile/test.py --batch_size 4
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o prof/nsys/Mol_dGnn_n_atoms_20 --force-overwrite true python Nsight_Profile/test.py --batch_size 4 --n_atoms 20
```

#### Profiling the Interaction via Pytorch Profiler
The following commands are example of runnning Pytorch Profiler to generate profile results for different model configurations:
```
    $ python Pytorch_Profile/test.py --batch_size 4
    $ python Pytorch_Profile/test.py --batch_size 4 --n_atoms 20
```
