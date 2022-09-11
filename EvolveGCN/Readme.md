EvolveGCN Profiling
=====

This repository contains the code for Dynamic Graph Neural Networks on Hardware: Bottleneck Analysis, published in IISWC 2022.

## Data

The dataset used in the paper:

- bitcoin Alpha: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
 
For downloaded data sets please place them in the 'data' folder.

## Requirements
  * PyTorch 1.0 or higher
  * Python 3.6

## Usage

Set --config_file with a yaml configuration file to run the experiments. For example:

To run EvolveGCN-O:
```sh
python run_exp.py --config_file ./experiments/parameters_bitcoin_alpha_edgecls_egcn_o.yaml
```
To run EvolveGCN-H:
```sh
python run_exp.py --config_file ./experiments/parameters_bitcoin_alpha_edgecls_egcn_h.yaml
```
The start_profiling option in the configuration file is used to control whether to profile. 

If use CPU, set use_cuda to false; if use GPU, set use_cuda to true.

The training/validation/testing dataset can all be used for profiling by setting the profile_part option to TRAIN/VALID/TEST.

If you want to profiling training, please set the profile_part option to TRAINING.
