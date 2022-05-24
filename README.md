This repo provides a collection of baselines for DGraphFin dataset. Please download the dataset from the [DGraph](http://dgraph.xinye.com) web and place it under the folder './dataset/DGraphFin/raw'.  

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch = 1.6.0  
- torch_geometric = 1.7.2  
- torch_scatter = 2.0.8  
- torch_sparse = 0.6.9  

- GPU: Tesla V100 32G  


## Training

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


## Results:
Performance on **DGraphFin**(10 runs):

| Methods   | Train AUC  | Valid AUC  | Test AUC  |
|  :----  | ----  |  ---- | ---- |
| MLP | 0.7221 ± 0.0014 | 0.7135 ± 0.0010 | 0.7192 ± 0.0009 |
| GCN | 0.7108 ± 0.0027 | 0.7078 ± 0.0027 | 0.7078 ± 0.0023 |
| GraphSAGE| 0.7682 ± 0.0014 | 0.7548 ± 0.0013 | 0.7621 ± 0.0017 |
| GraphSAGE (NeighborSampler)  | 0.7845 ± 0.0013 | 0.7674 ± 0.0005 | **0.7761 ± 0.0018** |
| GAT (NeighborSampler)        | 0.7396 ± 0.0018 | 0.7233 ± 0.0012 | 0.7333 ± 0.0024 |
| GATv2 (NeighborSampler)      | 0.7698 ± 0.0083 | 0.7526 ± 0.0089 | 0.7624 ± 0.0081 |