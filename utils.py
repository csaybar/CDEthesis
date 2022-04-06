import sklearn.model_selection
import torch
import numpy as np
from torch import nn
import rasterio as rio

from scipy.ndimage.filters import convolve
from skimage.morphology import disk, dilation


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                raise StopIteration("The loss does not decreased in %s epochs." % self.patience)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def split_dataset(iterator, test_size=0.2, val_size=0.2, random_state=42):
    # Create Xtrain and Xtest dataset
    Xtrain, Xtest = sklearn.model_selection.train_test_split(
        iterator, test_size=test_size, random_state=random_state
    )
    # Create Xtrain and Xval dataset
    n_val_size = val_size / (1 - test_size)
    Xtrain, Xval = sklearn.model_selection.train_test_split(
        Xtrain, test_size=n_val_size, random_state=random_state
    )
    return Xtrain, Xval, Xtest


class FilteredTverskyMetric(nn.Module):
    """ Soft Tversky loss used in cloudnet++
    
    Reference:
        Cloud and Cloud Shadow Segmentation for Remote
        Sensing Imagery via Filtered Jaccard Loss Function
        and Parametric Augmentation
        https://arxiv.org/pdf/2001.08768.pdf
    
    Examples:
        >>> y_true = torch.Tensor([0, 0, 0, 0, 1, 1])
        >>> y_pred = torch.Tensor([0, 0, 0, 1, 1, 1])
        >>> metrics = FilteredTverskyMetric()
        >>> metrics.forward(y_pred, y_true)
    """    
    def __init__(self, alpha=0.3, beta=0.7):
        super(FilteredTverskyMetric, self).__init__()
        self.KG = 1
        self.KJ = 1
        self.m = 1000
        self.pc = 0.5
        self.smooth = 1e-5
        self.alpha = alpha
        self.beta = beta
    def forward(self, y_pred, y_true):
        total_len = y_true.shape[0]
        total_one = y_true.sum().item()

        # Inverted Tversky (GL1)
        # True Positives, False Positives & False Negatives                
        y_true_complement = (y_true == False)*1
        y_pred_complement = 1 - y_pred
        TP = (y_pred_complement * y_true_complement).sum()        
        FP = ((1 - y_true_complement) * y_pred_complement).sum()        
        FN = (y_true_complement * (1 - y_pred_complement)).sum()
        GL = Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        # Inverted Tversky (JL)
        # True Positives, False Positives & False Negatives
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()
        JL = Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        FJL_01 = (self.KG * GL)/(1 + torch.exp(self.m*(sum(y_true) - self.pc)))
        FJL_02 = (self.KJ * JL)/(1 + torch.exp(self.m*(-sum(y_true) + self.pc)))
        return FJL_01 + FJL_02