# ASTGNN

This is a Pytorch implementation of ASTGNN. Now the corresponding paper is available online at https://ieeexplore.ieee.org/document/9346058.

This code contains a Pytorch implementation of ASTGNN and alterations for profiling ASTGNN.

# Requirements
```python
pip install requirements.txt
```

# Data
The dataset used for ASTGNN is obtained from the Caltrans Performance Measurement System (PeMS).

# Train and Test

We take the commands on PEMS04 for example.

Step 1: Process dataset:

```python
python prepareData.py --config configurations/PEMS04.conf
```

Step 2: train and test the model:

To test the model, comment train_main() and uncomment predict_main in train_ASTGNN.py.
```python
python train_ASTGNN.py --config configurations/PEMS04.conf
```

The settings for each experiments are given in the "configurations" folder.