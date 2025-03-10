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


