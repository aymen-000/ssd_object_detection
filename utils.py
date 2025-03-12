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


def find_jaccard_overlap(set1 , set2) : 
    """
     Find IOU of every comibination between set1 (set of bboxes) 

     Args : 
        set1 ,set2 : a tensor of dimension (n , 4) # n is the number of bboxes

    Output : 
            IOU of between every element 
    """
    n1 , n2 = set1.shape()[0] , set2.shape()[0]
    lower_bound_intersection = torch.zeros((n1 , n2 , 2))
    upper_bound_intersection = torch.zeros((n1 , n2 , 2))
    for i in range(n1) : 
        for j in range(n2) : 
            lower_bound_intersection[i , j ] = torch.max(set1[i , :2] , set2[j , :2])
            upper_bound_intersection[i , j] = torch.min(set1[i , 2:] , set2[j , 2:])
    intersection_dime = torch.clamp(upper_bound_intersection - lower_bound_intersection , min=0) 


    intersection =  intersection_dime[ : , : , 0] * intersection_dime[: , : , 1] 
    


    
