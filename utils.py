import torch.nn as nn 
import torch.nn.functional as F
import torch
import os
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
    Compute IoU (Jaccard index) between two sets of bounding boxes.

    Args:
        set1: Tensor of shape (n1, 4) in (x_min, y_min, x_max, y_max) format.
        set2: Tensor of shape (n2, 4) in (x_min, y_min, x_max, y_max) format.

    Returns:
        IoU matrix of shape (n1, n2)
    """
    # Calculate intersection
    lower_bounds = torch.max(set1[:, None, :2], set2[:, :2])  # (n1, n2, 2)
    upper_bounds = torch.min(set1[:, None, 2:], set2[:, 2:])  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # Avoid negative values
    intersection_area = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

    # Compute areas
    areas_set1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])  # (n1)
    areas_set2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])  # (n2)

    # Compute union
    union_area = areas_set1[:, None] + areas_set2 - intersection_area  # (n1, n2)

    # IoU computation
    return intersection_area / union_area  # (n1, n2)


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


def decay_lr(optimizer: torch.optim.Optimizer, decay_factor):
    """
    Scale learning rate by a specific factor.
    Args:
        optimizer: optimizer whose lr will be decayed
        decay_factor: scale factor for learning rate
    Return: None
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * decay_factor
    
    print(f"DECAY LEARNING RATE.\nTHE NEW LR IS: {optimizer.param_groups[0]['lr']}")


def save_checkpoints(epoch, model, optimizer):
    """
    Save model checkpoints.
    Args:
        epoch: epoch number
        model: model used
        optimizer: optimizer used
    Return: None
    """
    s = {
        "epochs": epoch,
        "model": model,
        "optimizer": optimizer
    }
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(s, filename)

class AverageMeter(object) : 
    def __init__(self):
        self.reset() 
    
    def reset(self) : 
        self.val = 0 
        self.avg = 0 
        self.sum  =0 
        self.count = 0 
    
    def update(self , val , n=1) : 
        self.val = val 
        self.sum += val*n
        self.count += n 
        self.avg = self.sum / self.count


def clip_gradient(optimizer:torch.optim.Optimizer , grad_clip ) : 
    """
        Clip  gradients computed during training to avoid explosion 

        Args : 
            optimizer : optim we used 
            grad_clip : clip value
    """
    for group in optimizer.param_groups : 
        for param in group['params'] : 
            if param.grad is not None : 
                param.grad.data.clam_(-grad_clip , grad_clip)


def load_model_pretrained_params(model, weight_file, optimizer=None, device='cuda'):
    """
    Load pretrained weights into a model.
    
    Args:
        model: The model to load weights into
        weight_file: Path to the checkpoint file containing weights
        optimizer: Optional optimizer to restore state
        device: Device to load the model on ('cuda' or 'cpu')
        
    Return:
        start_epoch: Epoch to resume from (0 if not resuming training)
        model: The model with loaded weights
        optimizer: The optimizer with loaded state (if provided)
    """
    if not os.path.isfile(weight_file):
        print(f"No checkpoint found at '{weight_file}'")
        return 0, model, optimizer
        
    print(f"Loading checkpoint from '{weight_file}'")
    checkpoint = torch.load(weight_file, map_location=device)
    
    # Handle checkpoint format from the provided code
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        start_epoch = 0
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"\nLoaded checkpoint from epoch {start_epoch - 1}.\n")
            
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer = checkpoint['optimizer']
    else:
        # If checkpoint is just weights
        model.load_state_dict(checkpoint)
        start_epoch = 0
        print("Loaded pretrained weights successfully")
    
    # Move to specified device
    model = model.to(device)
        
    return start_epoch, model, optimizer


def calc_mAP(det_boxes , det_labels , det_scores , true_boxes , true_labels , true_diificulties)  : 
    pass 