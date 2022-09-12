# ASTGNN Profiling

The following is the code for performing analysis on ASTGNN.

Link to the original paper:
https://ieeexplore.ieee.org/document/9346058

## Related Paper and Github Project

- [Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting](https://ieeexplore.ieee.org/document/9346058) [[Github](https://github.com/guoshnBJTU/ASTGNN)]


## Requirements
```python
pip install requirements.txt
```

## Data
The dataset used for ASTGNN is obtained from the Caltrans Performance Measurement System (PeMS).

## Train and Test
Set --config with a configuration file to train and test the model.

Sample commands on PEMS04 dataset:

Step 1: Preprocess the dataset:

```python
python prepareData.py --config configurations/PEMS04.conf
```

Step 2: train and test the model:

To test the model, comment train_main() and uncomment predict_main() in train_ASTGNN.py.
```python
python train_ASTGNN.py --config configurations/PEMS04.conf
```

The settings for each experiments are given in the "configurations" folder.
