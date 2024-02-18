import torch
import torch.nn as nn
import torch.nn.functional as F

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for multi-class classification
    """
    def focal_loss(input, target):
        BCE = F.cross_entropy(input, target, reduction='none')
        
        p = torch.exp(-BCE)
        loss = (1 - p) ** gamma * BCE
        
        return torch.mean(alpha * loss)
    return focal_loss

def weighted_categorical_cross_entropy(weights):
    """
    Weights should be a 1D Tensor assigning weight to each class.
    """
    weights = weights.float()
    
    def loss(input, target):
        log_probs = F.log_softmax(input, dim=-1)
        loss = -weights * target * log_probs  
        loss = loss.sum(-1)
        
        return loss.mean()
    
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseCategoricalFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        """
        input: [N, C]
        target: [N] 
        """
        
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        
        logpt = logpt.gather(1, target.view(-1,1))
        pt = pt.gather(1, target.view(-1,1))
                
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()
        
        return loss