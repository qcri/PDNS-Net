from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import torch
import numpy

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
    print('tn, fp, fn, tp', tn, fp, fn, tp)
    
    fpr = fp/(fp+tn)
    # print('SCORE', accuracy, f1, auc)
    return {'acc':accuracy, 'f1':f1, 'auc':auc, 'prec':prec, 'recall':recall, 'fpr':fpr}

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