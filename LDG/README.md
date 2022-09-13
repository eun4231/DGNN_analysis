## LDG: Latent Dynamic Graph, DyRep: Learning Representations over Dynamic Graphs

This page contains the code for the model DyRep and LDG and for performing analysis using two profiling tools: Pytorch Profiler and Nsight Systems.

---

## Related Paper and Github Project

- [Learning Temporal Attention in Dynamic Graphs with Bilinear Interactions](https://arxiv.org/abs/1909.10367) [[Github](https://github.com/uoguelph-mlrg/LDG)]

- [DyRep: Learning representations over dynamic graphs](https://openreview.net/forum?id=HyePrhR5KX)


## Dataset
- [Social Evolution](http://realitycommons.media.mit.edu/socialevolution4.html)

- [Github](https://www.gharchive.org/)



---
## Evaluate LDG, DyRep Inference

### Social Evolution

Running DyRep model on Social Evolution:

`python main.py --log_interval 300  --data_dir ./SocialEvolution/`.

Running LDG model with a learned graph, sparse prior and biliear interactions:

`python main.py --log_interval 300  --data_dir ./SocialEvolution/ --encoder mlp --soft_attn --bilinear --bilinear_enc --sparse`


### GitHub

To run Github experiments, use the same arguments, but add `--dataset github --data_dir ./Github`.

To use the Frequency bias, add the `--freq` flag.

