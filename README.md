<p align="center">
  <img height="100" src="https://raw.githubusercontent.com/DGraphXinye/dgraph_theme/main/img/DGraph_log.svg" />
</p>

--------------------------------------------------------------------------------
* [Pytorch Geometric Loader](#Pytorch-Geometric-Loader)
* [Baselines](#Baselines)
  * [Environments](#Environments)
  * [Training](#Training)
  * [Performances](#Performances)

# Pytorch Geometric Loader
[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (`>=2.2.0`) provides an easy-to-use dataset loader for [DGraphFin](https://dgraph.xinye.com/dataset). Below is an example in [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) with only a few lines of code to load [DGraphFin](https://dgraph.xinye.com/dataset) and get the train/valid/test mask.

```python
import torch_geometric

# check your torch_geometric version and make sure it is not lower than 2.2.0
print(torch_geometric.__version__)
>>> '2.2.0'

# Please download DGraphFin dataset file 'DGraphFin.zip' on our website 'https://dgraph.xinye.com' and place it under directory './dataset/raw'
# Otherwise an error would pop out "Dataset not found. Please download 'DGraphFin.zip' from 'https://dgraph.xinye.com' and move it to './raw' "
from torch_geometric.datasets import DGraphFin

dataset = DGraphFin(root='./dataset')
data = dataset[0]

data
>>> Data(x=[3700550, 17], edge_index=[2, 4300999], y=[3700550], edge_type=[4300999], edge_time=[4300999], train_mask=[3700550], val_mask=[3700550], test_mask=[3700550])
```

**Note:** Please download DGraphFin dataset file 'DGraphFin.zip' on our website 'https://dgraph.xinye.com' and place it under directory `'./dataset/raw'` before running the example, otherwise an error would pop out `"Dataset not found. Please download 'DGraphFin.zip' from 'https://dgraph.xinye.com' and move it to './raw' "`

# Baselines

This repo provides a collection of baselines of [DGraphFin](https://dgraph.xinye.com/dataset). Please download the dataset file on our [website](http://dgraph.xinye.com) and place it under the folder `'./dataset/DGraphFin/raw'`.  

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch = 1.6.0  
- torch_geometric = 1.7.2  
- torch_scatter = 2.0.8  
- torch_sparse = 0.6.9  

- GPU: Tesla V100 32G  


## Training
To get the performance for each model, simply run the following lines of code:

- **MLP**
```bash
python gnn.py --model mlp --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GCN**
```bash
python gnn.py --model gcn --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GraphSAGE**
```bash
python gnn.py --model sage --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GraphSAGE (NeighborSampler)**
```bash
python gnn_mini_batch.py --model sage_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GAT (NeighborSampler)**
```bash
python gnn_mini_batch.py --model gat_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```

- **GATv2 (NeighborSampler)**
```bash
python gnn_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
```


## Performances:
Below are the performances on **DGraphFin**(10 runs):

| Methods   | Train AUC  | Valid AUC  | Test AUC  |
|  :----  | ----  |  ---- | ---- |
| MLP | 0.7221 ± 0.0014 | 0.7135 ± 0.0010 | 0.7192 ± 0.0009 |
| GCN | 0.7108 ± 0.0027 | 0.7078 ± 0.0027 | 0.7078 ± 0.0023 |
| GraphSAGE| 0.7682 ± 0.0014 | 0.7548 ± 0.0013 | 0.7621 ± 0.0017 |
| GraphSAGE (NeighborSampler)  | 0.7845 ± 0.0013 | 0.7674 ± 0.0005 | **0.7761 ± 0.0018** |
| GAT (NeighborSampler)        | 0.7396 ± 0.0018 | 0.7233 ± 0.0012 | 0.7333 ± 0.0024 |
| GATv2 (NeighborSampler)      | 0.7698 ± 0.0083 | 0.7526 ± 0.0089 | 0.7624 ± 0.0081 |
