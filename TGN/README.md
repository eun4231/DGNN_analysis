## TGN: Temporal Graph Networks
This page contains the code for the model TGN and the two profiling tools: Pytorch Profiler and Nsight Systems.

---
## Model's Paper and its GitHub Page
- [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) [[GitHub](https://github.com/twitter-research/tgn)]

## Datasets Used for profiling this model
- [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
- [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)



## Requirements
- Dependencies (with python >= 3.7):
```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```
---



## Evaluate the model
### Preprocess the data
The dense `npy` format have been applied to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```
#### General flags

```{txt}
optional arguments:
  -d DATA, --data DATA         Data sources to use (wikipedia or reddit)
  --bs BS                      Batch size
  --prefix PREFIX              Prefix to name checkpoints and results
  --n_degree N_DEGREE          Number of neighbors to sample at each layer
  --n_head N_HEAD              Number of heads used in the attention layer
  --n_epoch N_EPOCH            Number of epochs
  --n_layer N_LAYER            Number of graph attention layers
  --lr LR                      Learning rate
  --patience                   Patience of the early stopping strategy
  --n_runs                     Number of runs (compute mean and std of results)
  --drop_out DROP_OUT          Dropout probability
  --gpu GPU                    Idx for the gpu to use
  --node_dim NODE_DIM          Dimensions of the node embedding
  --time_dim TIME_DIM          Dimensions of the time embedding
  --use_memory                 Whether to use a memory for the nodes
  --embedding_module           Type of the embedding module
  --message_function           Type of the message function
  --memory_updater             Type of the memory updater
  --aggregator                 Type of the message aggregator
  --memory_update_at_the_end   Whether to update the memory at the end or at the start of the batch
  --message_dim                Dimension of the messages
  --memory_dim                 Dimension of the memory
  --backprop_every             Number of batches to process before performing backpropagation
  --different_new_nodes        Whether to use different unseen nodes for validation and testing
  --uniform                    Whether to sample the temporal neighbors uniformly (or instead take the most recent ones)
  --randomize_features         Whether to randomize node features
  --dyrep                      Whether to run the model as DyRep
```


#### Model Inference

To run the inference of TGN model (tgn-attn), use the following command:
```
    # TGN-attn: Supervised learning on the wikipedia dataset
    $ python train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --gpu 0
    # TGN-attn: Self-supervised learning on the wikipedia dataset
    $ python train_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --gpu 0

    # TGN-attn-reddit: Supervised learning on the reddit dataset
    $ python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --gpu 0--n_runs 10
    # TGN-attn-reddit: Self-supervised learning on the reddit dataset
    $ python train_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --n_runs 10 --gpu 0
```



#### Profiling the Interaction via Nsight Systems
The following commands are example of runnning Nsight Systems to generate profile results for different model configurations:
- Self-supervised  learning:
    ```
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o prof/nsys/TGN_BS131072_edge_Pred --force-overwrite true python Nsight_Profile/train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --bs 4 --gpu 0 
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o prof/nsys/TGN_BS131072_edge_Pred --force-overwrite true python Nsight_Profile/train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --bs 4 --gpu 0 --n_degree 50
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o prof/nsys/TGN_BS131072_edge_Pred --force-overwrite true python Nsight_Profile/train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --bs 4 --gpu 0 --n_degree 50 --n_head 50
    ```

#### Profiling the Interaction via Pytorch Profiler
The following commands are example of runnning Pytorch Profiler to generate profile results for different model configurations:
- Self-supervised learning:
    ```
    $ python Pytorch_Profile/train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --bs 32 --gpu 0
    $ python Pytorch_Profile/train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --bs 32 --gpu 0 --n_degree 50
    $ python Pytorch_Profile/train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --n_runs 10 --bs 32 --gpu 0 --n_degree 50 --n_head 50
    ```