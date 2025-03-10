import torch.nn as nn 
import torch.nn.functional as F
import torch

def decimate(tensor , m) : 
    """
        Decimate tensor by factor m 

        Args : 
            tensor : tensor to be decimated 
            m : lsit of decimated factors 

        Return : 
            decimated tensor
    """
    assert len(m) == len(tensor.shape)
    slices = []
    for i , factor in enumerate(m) : 
        if factor is None : 
            slices.append(slice(None))
        else : 
            slices.append(slice(None , None , factor))
    print(slices)
    return tensor[slices]


import torch

def cxcy_to_xy(cxcy):
    """
    Convert center-form (cx, cy, w, h) bounding boxes to corner-form (x_min, y_min, x_max, y_max).
    
    Args:
        cxcy: Tensor of shape (n_boxes, 4), where each box is (cx, cy, w, h).
    
    Returns:
        Tensor of shape (n_boxes, 4), where each box is (x_min, y_min, x_max, y_max).
    """
    xy_min = cxcy[:, :2] - cxcy[:, 2:] / 2
    xy_max = cxcy[:, :2] + cxcy[:, 2:] / 2

    return torch.cat([xy_min, xy_max], dim=1) 


def pred_to_boxes(gcxgcy, priors):
    """
    Convert predicted bounding box coordinates (gcxgcy format) to absolute coordinates (cxcywh format)
    
    Args:
        gcxgcy: predicted bounding boxes in gcxgcy format, a tensor of size (n_priors, 4)
        priors: prior boxes with respect to which the predictions are made, a tensor of size (n_priors, 4)
        
    Returns:
        Converted bounding boxes in cxcywh format, a tensor of size (n_priors, 4)
    """
    return torch.cat([
        priors[:, :2] + gcxgcy[:, :2] * priors[:, 2:] / 10,  # c_x, c_y
        priors[:, 2:] * torch.exp(gcxgcy[:, 2:] / 5)         # w, h
    ], dim=1)



