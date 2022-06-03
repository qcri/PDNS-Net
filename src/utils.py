from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def score(pred, labels):
    tp, fp, tn, fn = confusion(labels, pred)
    print('tn, fp, fn, tp', tn, fp, fn, tp)
    
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    accuracy = (pred == labels).sum() / len(pred)
    micro_f1 = f1_score(labels, pred, average='micro')
    macro_f1 = f1_score(labels, pred, average='macro')
    try:
        f1 = f1_score(labels, pred, average='weighted')
    except:
        print('Exception occurred while calculating F1 Score', labels, pred)
        f1 = 0
    auc = roc_auc_score(labels, pred)
    prec, recall = precision_score(labels, pred), recall_score(labels,pred)
    
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    # print('tn, fp, fn, tp', tn, fp, fn, tp)
    
    fpr = fp/(fp+tn)
    # print('SCORE', accuracy, f1, auc)
    return {'acc':accuracy, 'f1':f1, 'auc':auc, 'prec':prec, 'recall':recall, 'fpr':fpr, 'mi_f1':micro_f1, 'ma_f1':macro_f1}


def confusion(truth, prediction):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def to_homogeneous(data, transform=None) -> Data:
        homo_data = data.clone()

        features_shape = sum([node_features.shape[1] for _, node_features in homo_data.x_dict.items()])
        mask_types = ['train_mask', 'val_mask']
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
                y.append(torch.full((node_features.shape[0],),2))
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

    
def plot_degree_dist_dom_ip_log(g, node_type, title=""):
    degrees = nx.degree(g)
    degree_df = pd.DataFrame(degrees, columns=['node_id', 'degree']).sort_values('node_id', ascending=True) 
    degree_df['node_type'] = node_type
    degree_df = degree_df.sort_values('degree', ascending=False)
    degree_df = degree_df[degree_df.degree > 0]
    
    domain_df = degree_df[degree_df.node_type == 0]
    ip_df = degree_df[degree_df.node_type == 1]
    
    dom_degrees = domain_df.degree.values
    ip_degrees = ip_df.degree.values
    max_deg = max(dom_degrees[0], ip_degrees[0])

    plt.figure(figsize=(4, 3))
    [[dom_counts, ip_counts],bins,_]=plt.hist([dom_degrees, ip_degrees],bins=max_deg)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.close()

    dom_countsnozero=dom_counts*1.
    dom_countsnozero[dom_counts==0]=-np.Inf
    ip_countsnozero=ip_counts*1.
    ip_countsnozero[ip_counts==0]=-np.Inf

    plt.figure(figsize=(3, 2.3))
    plt.scatter(bins[:-1],dom_countsnozero/float(sum(dom_counts)),s=10,marker='x',label='Domain')
    plt.scatter(bins[:-1],ip_countsnozero/float(sum(ip_counts)),s=10,marker='x',label="IP")
    plt.yscale('log'), plt.xscale('log')
    plt.xlabel('Degree (log)')
    plt.ylabel("Fraction of nodes (log)")
    plt.legend()
    plt.show()
    

def plot_degree_dist_labels_log(g, node_type, labels, title=""):
    degrees = nx.degree(g, nbunch=list(torch.nonzero(node_type == 0).t()[0].numpy()))
    degree_df = pd.DataFrame(degrees, columns=['node_id', 'degree']).sort_values('node_id', ascending=True) 
    degree_df['label'] = labels
    degree_df = degree_df.sort_values('degree', ascending=False)
    degree_df = degree_df[degree_df.degree > 0]

    ben_df = degree_df[degree_df.label == 0]
    mal_df = degree_df[degree_df.label == 1]
    
    ben_degrees = ben_df.degree.values
    mal_degrees = mal_df.degree.values
    max_deg = max(mal_degrees[0], ben_degrees[0])

    plt.figure(figsize=(4, 3))
    [[ben_counts, mal_counts],bins,_]=plt.hist([ben_degrees, mal_degrees],bins=max_deg)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.close()

    ben_countsnozero=ben_counts*1.
    ben_countsnozero[ben_counts==0]=-np.Inf
    mal_countsnozero=mal_counts*1.
    mal_countsnozero[mal_counts==0]=-np.Inf

    plt.figure(figsize=(3, 2.3))
    plt.scatter(bins[:-1],ben_countsnozero/float(sum(ben_counts)),s=10,marker='x',label='Benign',color='g')
    plt.scatter(bins[:-1],mal_countsnozero/float(sum(mal_counts)),s=10,marker='x',label="Malicious",color='r')
    plt.yscale('log'), plt.xscale('log')
    plt.xlabel('Degree (log)')
    plt.ylabel("Fraction of nodes (log)")
    plt.legend()
    plt.show()
