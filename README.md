# SSF-Strategic-Selection-and-Forgetting

This is the code for the paper: *Continual Learning for Network Intrusion Detection with MMD-Based Drift Detection*.

Bo Xu, Qiang Yang, Tao Zhang, Xu Tong, Fan Yang, Rui Shi.

## Dependencies

The project is implemented using PyTorch and has been tested on the following hardware and software configuration:

- Ubuntu 22.04
- NVIDIA GPU (32 GB vGPU in our experiments)
- CUDA 12.1
- PyTorch 2.1.0
- Python 3.10

### Installation

To install the necessary libraries and dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Experiments

We tested the effectiveness of our proposed method on the NSL-KDD and UNSW-NB15 datasets. Preprocessed versions of these datasets are provided in this repository, allowing for immediate execution. The continuous attributes have been normalized, and categorical attributes have been one-hot encoded.

Here are two examples for each dataset (NSL-KDD and UNSW-NB15) of how to start training:

```bash
python cl_nids_mmd.py --dataset nsl --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --new_sample_weight 3 --drift_threshold 0.001
```

```bash
python cl_nids_mmd.py --dataset unsw --epochs 200 --epoch_1 180 --sample_interval 20000  --num_labeled_sample 200 --new_sample_weight 60 --drift_threshold 0.001
```
