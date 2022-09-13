## ASTGNN: Attention based Spatial-Temporal Graph Neural Network

This page contains the code for the model ASTGNN and for performing analysis using two profiling tools: Pytorch Profiler and Nsight Systems.

---
## Related Paper and Github Project

- [Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting](https://ieeexplore.ieee.org/document/9346058) [[Github](https://github.com/guoshnBJTU/ASTGNN)]


## Dataset
- [Performance Measurement System (PeMS)](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source)

## Requirements
```python
pip install requirements.txt
```
---

## Evaluate ASTGNN Inference
Set --config with a configuration file to train and test the model.

Sample commands on PEMS04 dataset:

#### Step 1: Preprocess the dataset:

```python
python prepareData.py --config configurations/PEMS04.conf
```

#### Step 2: train and test the model:

To test the model, comment train_main() and uncomment predict_main() in train_ASTGNN.py.
```python
python train_ASTGNN.py --config configurations/PEMS04.conf
```

The settings for each experiments are given in the "configurations" folder.
