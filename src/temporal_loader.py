import numpy as np
from typing import Optional, Callable
from collections import defaultdict

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
    
    def fetch_node_info(self):
        domains = pd.read_csv(os.path.join(self.root, 'domains.csv')).sort_values('domain_node')
        ips = pd.read_csv(os.path.join(self.root, 'ips.csv')).sort_values('ip_node')

        return domains, ips
    
    def to_static(self, start=0, end=None):
        edges = {k: torch.empty((2, 0), dtype=torch.int)for k in self.data[0].edge_index_dict.keys()}

        for item in self.data[start:end]:
            for et, et_edges in item.edge_index_dict.items():
                edges[et] = torch.unique(torch.cat([edges[et], et_edges], dim=1), dim=1)

        out = self.data[0].clone()
        for et, et_edges in edges.items():
            del out[et]
            out[et].edge_index = et_edges
        return out

