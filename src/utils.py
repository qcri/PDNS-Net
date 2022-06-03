from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import torch
import numpy as np
from torch_geometric.data import Data


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