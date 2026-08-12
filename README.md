# CIKM26-X-Fed

Official repo for *Federated Personalization of Early-Exit Networks*, which is accepted by CIKM 2026.

## Usage
### Guidance
All baselines are implemented in the directory of `trainer/alg`.

All baselines extends Client and Server in `trainer/alg/eefl.py`, which extends that of `trainer/base.py`.

The hyperparameter `mode` determines whether we train the last exit or jointly train all exits.

### Reproduce our results
1. Open the directory of `script`

2. Run the following scripts

```
bash run_cifar10.sh
bash run_cifar100.sh
bash run_tiny.sh
bash run_agnews.sh
```

## Results
TBD


## Citation
If you find X-Fed useful or relevant to your reserach, please kindly cite our paper:
```
@inproceedings{liu2026federated,
  title     = {Federated Personalization of Early-Exit Networks},
  author    = {Boyi Liu and Zimu Zhou and Cheng Fang and Yongxin Tong},
  booktitle = {CIKM},
  year      = {2026}
}
```

## Acknowledgements
The data partitioning module is adopted from [PFLlib](https://github.com/TsingZ0/PFLlib).

