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


def find_jaccard_overlap(set1, set2):
    """
    Find IoU (Intersection over Union) of every combination between set1 and set2 (sets of bboxes)
    
    Args:
        set1, set2: tensors of dimension (n, 4) where n is the number of bboxes
                    Format: each box is [x_min, y_min, x_max, y_max]
    
    Output:
        IoU between every element as a tensor of shape (n1, n2)
    """
    # Get number of boxes in each set
    n1, n2 = set1.shape[0], set2.shape[0]
    
    # Initialize tensors for intersection bounds
    lower_bound_intersection = torch.zeros((n1, n2, 2))
    upper_bound_intersection = torch.zeros((n1, n2, 2))
    
    # Find intersection bounds
    for i in range(n1):
        for j in range(n2):
            lower_bound_intersection[i, j] = torch.max(set1[i, :2], set2[j, :2])
            upper_bound_intersection[i, j] = torch.min(set1[i, 2:], set2[j, 2:])
    
    # Calculate intersection areas
    intersection_dims = torch.clamp(upper_bound_intersection - lower_bound_intersection, min=0)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]
    
    # Find the area of each box in both sets
    dx_set1 = set1[:, 2] - set1[:, 0]
    dy_set1 = set1[:, 3] - set1[:, 1]
    dx_set2 = set2[:, 2] - set2[:, 0]
    dy_set2 = set2[:, 3] - set2[:, 1]
    
    areas_set1 = dx_set1 * dy_set1
    areas_set2 = dx_set2 * dy_set2
    
    # Calculate union areas
    union = areas_set1.unsqueeze(1) + areas_set2.unsqueeze(0) - intersection
    
    # Return IoU
    return intersection / union


def non_maximum_suppression(boxes: torch.Tensor, device, iou: torch.Tensor, max_overlap: float):
    """
    Perform Non-Maximum Suppression (NMS) on a set of bounding boxes.

    :param boxes: Tensor of shape (n, 4) containing bounding box coordinates.
    :param device: The device to use (CPU or GPU).
    :param iou: Precomputed IoU matrix of shape (n, n).
    :param max_overlap: IoU threshold for suppression.
    :return: Indices of the boxes to keep.
    """
    n = boxes.shape[0]
    suppress = torch.zeros(n, dtype=torch.bool, device=device) 
    
    for i in range(n):
        if suppress[i]:  # If already suppressed, skip
            continue
        
        # Vectorized suppression: mark all boxes with IoU > max_overlap
        suppress |= (iou[i] > max_overlap)

        # But don't suppress the current box itself
        suppress[i] = False  

    # Return indices of boxes that were not suppressed
    keep = torch.where(~suppress)[0]
    return keep