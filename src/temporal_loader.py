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
                 pre_transform: Optional[Callable] = None, alt_domain=None):
        super().__init__(root, transform, pre_transform)
        processed = self._load_dataset(root, alt_domain)
        self.data, self.slices = self.collate([processed])
        
    @property
    def num_classes(self) -> int:
        return 2
    
    def _load_dataset(self, root, alt_domain=None):
        edge_type_nodes = {
            'apex': ['domain_node', 'domain_node'],
            'similar': ['domain_node', 'domain_node'],
            'resolves': ['domain_node', 'ip_node']
        }

        directory = [os.path.join(root, f) for f in os.listdir(root)]
        edge_files = [f for f in directory if os.path.isfile(f) and 'timestamp' in f]
        edge_files = sorted(edge_files, key=lambda x: int(x.split("/")[-1].split("_")[1]))

        domains_file = 'domains.csv' if alt_domain is None else alt_domain
        domains = pd.read_csv(os.path.join(root, domains_file)).sort_values('domain_node')
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
    
    @staticmethod
    def _remove_isolated(data):
        domains = set()

        for (src_type, _, dst_type), edges in data.edge_index_dict.items():
            if src_type == 'domain_node':
                domains.update(edges[0].numpy())
            if dst_type == 'domain_node':
                domains.update(edges[1].numpy())

        out = data.clone()
        out['domain_node'].y = torch.tensor([
            2 if i not in domains else v for i, v in enumerate(data['domain_node'].y)
        ]).t()
        return out, domains

    @staticmethod
    def _label_mask(labels, balance_gt=False, verbose=False):
        labeled = labels < 2

        labeled_indices = labeled.nonzero()
        if balance_gt:
            mal_nodes = (labels == 1).nonzero()
            ben_nodes = (labels == 0).nonzero()
            min_count = min(len(mal_nodes), len(ben_nodes))

            mal_nodes = mal_nodes[torch.randperm(len(mal_nodes))[:min_count]]
            ben_nodes = ben_nodes[torch.randperm(len(ben_nodes))[:min_count]]
            labeled_indices = np.concatenate((mal_nodes, ben_nodes))
            print('Balanced labeled count:', len(labeled_indices))

        mask = torch.zeros(len(labeled), dtype=torch.bool)
        mask[labeled_indices] = True

        return mask
    
    def train_test_split(self, train=slice(0, 5), test_list=[6], balance_gt = True, verbose=True):
        domain_timestamps = self.fetch_node_info()[0].set_index('domain_node').timestamp

        train_data = self.to_static(train.start, train.stop) 
        train_data, domains = DNS._remove_isolated(train_data)

        print('Train labelled node count:', len((train_data['domain_node'].y < 2).nonzero()))
        train_data['domain_node']['train_mask'] = DNS._label_mask(train_data['domain_node'].y, balance_gt, verbose)
        train_data['domain_node']['val_mask'] = torch.zeros(len(train_data['domain_node'].y), dtype=torch.bool)

        test_data_list = []
        for test in test_list:
            test_data = self.to_static(train.start, test) 

            test_domains = domain_timestamps[domain_timestamps == test]        
            test_data['domain_node'].y = torch.tensor([
                2 if i not in test_domains.index else v for i, v in enumerate(test_data['domain_node'].y)
            ]).t()

            print(f'Test {test} labelled node count:', len((test_data['domain_node'].y < 2).nonzero()))
            test_data['domain_node']['val_mask'] = DNS._label_mask(test_data['domain_node'].y, balance_gt, verbose)
            test_data['domain_node']['train_mask'] = torch.zeros(len(test_data['domain_node'].y), dtype=torch.bool)
            test_data_list.append(test_data)    

        return train_data, test_data_list