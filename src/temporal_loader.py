import numpy as np
from typing import Optional, Callable

import pandas as pd
import os

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric_temporal.signal import DynamicHeteroGraphStaticSignal


class DNS(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        processed = self._load_dataset(root)
        self.data, self.slices = self.collate([processed])
        
    @property
    def num_classes(self) -> int:
        return 2
    
    def _load_dataset(self, root):
        edge_type_nodes = {
            'apex': ['domain_node', 'domain_node'],
            'similar': ['domain_node', 'domain_node'],
            'resolves': ['domain_node', 'ip_node']
        }

        directory = [os.path.join(root, f) for f in os.listdir(root)]
        edge_files = [f for f in directory if os.path.isfile(f) and 'timestamp' in f]
        edge_files = sorted(edge_files, key=lambda x: int(x.split("/")[-1].split("_")[1]))

        domains = pd.read_csv(os.path.join(root, 'domains.csv')).sort_values('domain_node')
        label_dict = {
            'domain_node': domains['type'].apply(lambda x: 1 if x == 'malicious' else (0 if x == 'benign' else 2)).values
        }

        edge_indices = []
        edge_weights = []
        targets = []
        for edge_file in edge_files:
            all_edges = pd.read_csv(edge_file)

            timestamp_edges = {}
            for edge_type, edges in all_edges.groupby('type'):
                src_type, dst_type = edge_type_nodes[edge_type]
                key = (src_type, edge_type, dst_type)
                timestamp_edges[key] = edges.loc[:, ['source', 'target']].astype(int).values.T
            edge_indices.append(timestamp_edges)
            edge_weights.append(None)
            targets.append(label_dict)

        domain_features = pd.read_csv(os.path.join(root, 'domain_features.csv')).sort_values('domain_node')
        ip_features = pd.read_csv(os.path.join(root, 'ip_features.csv')).sort_values('ip_node')

        feature_dict = {
            'domain_node': domain_features.set_index('domain_node').values,
            'ip_node': ip_features.set_index('ip_node').values
        }

        graph = DynamicHeteroGraphStaticSignal(edge_indices, edge_weights, feature_dict, targets)
        return graph