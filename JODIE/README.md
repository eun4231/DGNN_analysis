## JODIE: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks
This page contains the code for the model JODIE and the two profiling tools: Pytorch Profiler and Nsight Systems.

---
## Model's Paper and its GitHub Page
- [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://arxiv.org/abs/1908.01207) [[GitHub](https://github.com/srijankr/jodie)]

## Datasets Used for profiling this model
- [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
- [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)
- [LastFM](http://snap.stanford.edu/jodie/lastfm.csv)



### Requirements
- Python >= 3.7
```{bash}
numpy==1.22.0
gpustat==0.5.0
tqdm==4.32.1
torch==0.4.1
scikit_learn==0.19.1
```
---

## Evaluate JODIE Inference

#### General flags

This code can be given the following command-line arguments:
1. `--network`: this is the name of the file which has the data in the `data/` directory. The file should be named `<network>.csv`. The dataset format is explained below. This is a required argument. 
2. `--model`: this is the name of the model and the file where the model will be saved in the `saved_models/` directory. Default value: jodie.
3. `--gpu`: this is the id of the gpu where the model is run. Default value: -1 (to run on the GPU with the most free memory).
4. `--epochs`: this is the maximum number of interactions to train the model. Default value: 50.
5. `--embedding_dim`: this is the number of dimensions of the dynamic embedding. Default value: 128.
6. `--train_proportion`: this is the fraction of interactions (from the beginning) that are used for training. The next 10% are used for validation and the next 10% for testing. Default value: 0.8
7. `--state_change`: this is a boolean input indicating if the training is done with state change prediction along with interaction prediction. Default value: True.

##### Interaction prediction

To run the inference of the model for the interaction prediction task, use the following command:
```
    $ python evaluate_interaction_prediction.py --network <network> --model jodie --epoch 1
```

##### State change prediction

To run the inference of the model for the state change prediction task, use the following command:
```
   $ python evaluate_state_change_prediction.py --network <network> --model jodie --epoch 1
```

#### Profiling the Interaction prediction running Nsight Systems
The following commands are example of runnning Nsight Systems to generate a profile result file:
- Sequential:
    ```
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o resProf/nsys/seq/interaction_seq_wikipedia --force-overwrite true python Nsight_Profile/evaluate_interaction_prediction_nsys.py --network wikipedia --model jodie --epoch 1 --gpu 0
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o resProf/nsys/seq/interaction_seq_reddit --force-overwrite true python Nsight_Profile/evaluate_interaction_prediction_nsys.py --network reddit --model jodie --epoch 1 --gpu 0
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o resProf/nsys/seq/interaction_seq_lastfm --force-overwrite true python Nsight_Profile/evaluate_interaction_prediction_nsys.py --network lastfm --model jodie --epoch 1 --gpu 0
    ```

- T-Batch:
    ```
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o resProf/nsys/tBatch/interaction_tBatch_warpUp_wikipedia --force-overwrite true python Nsight_Profile/evaluate_interaction_prediction_nsys_tBatch.py --network wikipedia --model jodie --epoch 1 --gpu 0
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o resProf/nsys/tBatch/interaction_tBatch_warpUp_reddit --force-overwrite true python Nsight_Profile/evaluate_interaction_prediction_nsys_tBatch.py --network reddit --model jodie --epoch 1 --gpu 0
    $ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --cuda-memory-usage=true -o resProf/nsys/tBatch/interaction_tBatch_warpUp_lastfm --force-overwrite true python Nsight_Profile/evaluate_interaction_prediction_nsys_tBatch.py --network lastfm --model jodie --epoch 1 --gpu 0
    ```



#### Profiling the Interaction prediction running Pytorch Profiler
The following commands are example of runnning Pytorch Profiler to generate profile results for different model configurations:
- Sequential:

    ```
    $ python Pytorch_Profile/evaluate_interaction_prediction_prof.py --network wikipedia --model jodie --epoch 0
    $ python Pytorch_Profile/evaluate_interaction_prediction_prof_.py --network reddit --model jodie --epoch 0
    $ python Pytorch_Profile/evaluate_interaction_prediction_prof.py --network lastfm --model jodie --epoch 0
    ```
- T-Batch:
    ```
    $ python Pytorch_Profile/evaluate_interaction_prediction_prof_tBatch.py --network wikipedia --model jodie --epoch 0
    $ python Pytorch_Profile/evaluate_interaction_prediction_prof_tBatch.py --network reddit --model jodie --epoch 0
    $ python Pytorch_Profile/evaluate_interaction_prediction_prof_tBatch.py --network lastfm --model jodie --epoch 0

    ```