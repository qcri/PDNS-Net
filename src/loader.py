import numpy as np
from typing import Optional, Callable

import pandas as pd
import os
import json
from sklearn import preprocessing

import torch
from torch_geometric.data import (InMemoryDataset, HeteroData, Data)
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index


class DNS(InMemoryDataset):
    def __init__(self, root: str, is_multigraph=False, num_test=0.3, num_val=0.2, balance_gt=False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        processed = self._load_dataset(root, is_multigraph, num_test=num_test, num_val=num_val, balance_gt=balance_gt)
        self.data, self.slices = self.collate([processed])
        
    @property
    def num_classes(self) -> int:
        return int(self.data['domain_node'].y.max()) + 1
    
    def _load_dataset(self, root, is_multigraph, num_test=0.3, num_val=0.2, balance_gt=False):
        graph_data = load_graph(root, not_multigraph=not is_multigraph)
        graph_nodes, edges, has_public, has_isolates, pruning, extras = graph_data
        edge_type_nodes = {
            'apex': ['domain', 'domain'],
            'similar': ['domain', 'domain'],
            'resolves': ['domain', 'ip']
        }

        data = HeteroData()

        node_ilocs = {}
        for node_type, node_features in graph_nodes.items():
            node_type = node_type + "_node"
            for x, idx in enumerate(node_features.index):
                node_ilocs[idx] = (x, node_type)

            data[node_type].num_nodes = node_features.shape[0]
            data[node_type].x = torch.from_numpy(node_features.values).float()
            if node_type == 'domain_node':
                extras['node_iloc'] = extras['node_id'].apply(lambda x: node_ilocs[x][0] if x in node_ilocs else None)
                extras = extras.dropna()
                
                labels = extras.sort_values('node_iloc')['type'].apply(lambda x: 1 if x == 'malicious' else (0 if x == 'benign' else 2))
                
                data[node_type].y = torch.from_numpy(labels.values)
                labeled = labels.values < 2
                labeled_indices = labeled.nonzero()[0]
                
                # balance benign and mal nodes
                if balance_gt:
                    mal_nodes = (labels.values == 1).nonzero()[0]
                    ben_nodes = (labels.values == 0).nonzero()[0]

                    min_count = min(len(mal_nodes), len(ben_nodes))
                    perm = torch.randperm(min_count)

                    mal_nodes = mal_nodes[torch.randperm(len(mal_nodes))[:min_count]]
                    ben_nodes = ben_nodes[torch.randperm(len(ben_nodes))[:min_count]]
                    labeled_indices = np.concatenate((mal_nodes, ben_nodes))
                
                n_nodes = len(labeled_indices)
                perm = torch.randperm(n_nodes)

                test_idx = labeled_indices[perm[:int(n_nodes * num_test)]]
                val_idx = labeled_indices[perm[int(n_nodes * num_test):int(n_nodes * (num_test + num_val))]]
                train_idx = labeled_indices[perm[int(n_nodes * (num_test + num_val)):]]

                for v, idx in [('train', train_idx), ('test', test_idx), ('val', val_idx)]:
                    mask = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
                    mask[idx] = True
                    data[node_type][f'{v}_mask'] = mask
        
        for edge_type, edge_data in edges.groupby('type'):
            from_type = edge_data['source'].apply(lambda x: node_ilocs[x][1]).drop_duplicates().values[0]
            to_type = edge_data['target'].apply(lambda x: node_ilocs[x][1]).drop_duplicates().values[0]

            edge_data['source'] = edge_data['source'].apply(lambda x: node_ilocs[x][0])
            edge_data['target'] = edge_data['target'].apply(lambda x: node_ilocs[x][0])
            edge_data = torch.from_numpy(edge_data.loc[:, ['source', 'target']].values.T)

            data[from_type, edge_type, to_type].edge_index = edge_data
            
        return data
    
    def to_homogeneous(self, transform=None) -> Data:
        homo_data = self.data.clone()

        features_shape = sum([node_features.shape[1] for _, node_features in homo_data.x_dict.items()])
        mask_types = ['train_mask', 'val_mask', 'test_mask']
        masks = {k: [] for k in mask_types}
        y = []
        
        node_map = {node_type: i for i, node_type in enumerate(homo_data.node_types)}
        edge_map = {i: (node_map[edge_type[0]], node_map[edge_type[2]]) for i, edge_type in enumerate(homo_data.edge_types)}
        
        l_padding = 0
        for node_type, node_features in homo_data.x_dict.items():
            node_features = node_features.numpy()
            r_padding = features_shape - node_features.shape[1] - l_padding
            features = []
            for node_feature in node_features:
                resized = np.pad(node_feature, (l_padding, r_padding), 'constant', constant_values=(0, 0))
                features.append(resized)

            if 'y' in homo_data[node_type]:
                y.append(homo_data[node_type].y)
                for mask_type in mask_types:
                    masks[mask_type].append(homo_data[node_type][mask_type])
            else:
                y.append(torch.zeros(node_features.shape[0], dtype=torch.bool))
                for mask_type in mask_types:
                    masks[mask_type].append(torch.zeros(node_features.shape[0], dtype=torch.bool))

            l_padding += node_features.shape[1]
            homo_data[node_type].x =  torch.from_numpy(np.array(features)).float()

        homo_data = homo_data.to_homogeneous(add_edge_type=True, add_node_type=True)

        for mask_type, mask in masks.items():
            homo_data[mask_type] = torch.cat(mask)

        homo_data.y = torch.cat(y)
        homo_data.edge_map = edge_map
        num_nodes = homo_data.num_nodes
        
        if transform is not None:
            transform(homo_data)
        homo_data.num_nodes = num_nodes
            
        return homo_data
    
    @staticmethod
    def get_edge_map(data):
        if 'edge_map' in data:
            return [(
                str(edge_type),
                (data.node_type == edge_node_types[0]).nonzero().view(1, -1).cpu().numpy()[0],
                (data.node_type == edge_node_types[1]).nonzero().view(1, -1).cpu().numpy()[0]
            ) for edge_type, edge_node_types in data.edge_map.items()]
        else:
            return None
    

def load_graph(path, not_multigraph=True):
    """
    Load stored graph in the given directory 

    :param path: Directory of the stored graph
    :param not_multigraph: If set true, multigraph edges are merged (parallel edge between same adjecnodes)
    :returns: tuple of (nodes, edges, has public domains, has isolates, pruning info, extras
    """
    with open(os.path.join(path, 'summary.json'), 'r') as json_data:
        summary = json.load(json_data)

    has_public = summary['has_public']
    has_isolates = summary['has_isolates']
    pruning = summary['pruning']

    directory = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in directory if os.path.isfile(f)]

    edges = [f for f in files if 'edges' in f]
    if len(edges) > 0:
        edges = pd.read_csv(edges[0])
    else:
        raise Exception("No 'edges.csv' file found in the path")

    if not_multigraph:
        edges_sorted = pd.DataFrame(np.sort(edges.loc[:, ['source', 'target']].values, axis=1), columns=['source', 'target'])
        edges_sorted['type'] = edges.type

        edges_sorted = edges_sorted.sort_values(['source', 'target', 'type'])
        print(f"Remove parallel edges: {edges_sorted[edges_sorted.duplicated(['source', 'target'])].value_counts('type')}")
        edges = edges_sorted.drop_duplicates(['source', 'target'])

    graph_nodes = {}
    nodes = [f for f in files if 'nodes' in f]
    if len(nodes) > 0:
        for n_type_file in nodes:
            n_type = n_type_file.split('.')[-2]
            nodes_df = pd.read_csv(n_type_file, index_col=0)
            graph_nodes[n_type] = nodes_df
    else:
        raise Exception("No 'nodes.<node_type>.csv' files found in the path")

    extras = [f for f in files if 'extras' in f]
    if len(extras) > 0:
        extras = pd.read_csv(extras[0], index_col=0)
    else:
        extras = None

    return graph_nodes, edges, has_public, has_isolates, pruning, extras
