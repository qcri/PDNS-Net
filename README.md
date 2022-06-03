# PDNS-Net

Passive DNS Dataset of Domain Resolutions

## Repository Structure

Dataset, data loaders, statistics and experiments of the Passive DNS Dataset available in following file and directories.

* `data/DNS_2m` - Raw data of PDNS-Net 2 months graph
* `src` - Data loaders and utilities for experiments
* `experiments` - Notebooks containing various GNN experiments on the dataset
* `PyG Dataset Temporal.ipynb` - Notebook with an example code to load the dataset
* `Statistics.ipynb` - Statistics of the dataset presented in the paper


## Citation

Please check the following publication for details:
```
@article{pdnsnet,
  title={PDNS-Net: A Large Heterogeneous Graph Benchmark Dataset of Network Resolutions for Graph Learning},
  author={Kumarasinghe, Udesh and Deniz, Fatih and Nabeel, Mohamed},
  journal={arXiv preprint arXiv:2203.07969},
  year={2022}
}
```
