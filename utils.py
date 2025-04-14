import torch.nn as nn 
import torch.nn.functional as F
import torch
import os
from configuration import *
import torchvision.transforms.functional as FT
import random
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


def save_checkpoints(epoch, model, optimizer , filename):
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
    fil= f'{filename}.pth.tar'
    torch.save(s, file)

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


def calc_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, label_map, rev_label_map):
    """
    Calculate the Mean Average Precision (mAP) of detected objects 

    Args:
        det_boxes: List of tensors, detected bounding boxes for each image
        det_labels: List of tensors, detected labels for each image
        det_scores: List of tensors, confidence scores for each detection
        true_boxes: List of tensors, ground truth bounding boxes for each image
        true_labels: List of tensors, ground truth labels for each image
        true_difficulties: List of tensors, difficulty flags for each ground truth object
        label_map: Dictionary mapping label names to indices
        rev_label_map: Dictionary mapping indices to label names

    Returns:
        average_precisions: Dictionary of AP for all classes
        mean_average_precision: mAP value
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)
    n_classes = len(label_map)

    # Prepare data
    images_index = []
    for i in range(len(true_boxes)):
        images_index.extend([i] * true_labels[i].size(0))
        # Number of real objects in each image
    
    det_images = []
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
        # Number of detected objects in each image
    
    # Convert to tensors
    images_index = torch.LongTensor(images_index).to(DEVICE)
    det_images = torch.LongTensor(det_images).to(DEVICE)
    
    # Concatenate all tensors
    det_boxes = torch.cat(det_boxes, dim=0)
    det_labels = torch.cat(det_labels, dim=0)
    det_scores = torch.cat(det_scores, dim=0)
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    true_difficulties = torch.cat(true_difficulties, dim=0)

    # Initialize AP tensor
    AP = torch.zeros(n_classes - 1, dtype=torch.float).to(DEVICE)
    
    # Calculate AP for each class
    for c in range(1, n_classes):
        # Extract objects with class c
        class_images = images_index[true_labels == c]
        class_boxes = true_boxes[true_labels == c]
        class_difficulties = true_difficulties[true_labels == c]
        
        # Count number of easy (non-difficult) objects
        easy_class_obj = (1 - class_difficulties).sum().item()
        
        # Extract detections with class c
        det_class_images = det_images[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)
        
        # Initialize array to keep track of detected boxes
        class_boxes_detected = torch.zeros(class_boxes.size(0)).to(DEVICE)
        
        if n_class_detections == 0:
            continue
        
        # Sort detections by confidence score
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]
        
        # Initialize true positives and false positives tensors
        TP = torch.zeros(n_class_detections, dtype=torch.float).to(DEVICE)
        FP = torch.zeros(n_class_detections, dtype=torch.float).to(DEVICE)
        
        # Check each detection
        for d in range(n_class_detections):
            this_box_detection = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]
            
            # Find ground truth boxes for this image
            t_object_boxes = class_boxes[class_images == this_image]
            t_object_diff = class_difficulties[class_images == this_image]
            
            if t_object_boxes.size(0) == 0:
                # No ground truth boxes in this image
                FP[d] = 1
                continue
            
            # Calculate IoU with ground truth boxes
            overlaps = find_jaccard_overlap(this_box_detection, t_object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)
            
            # Get original index
            original_ind = torch.LongTensor(range(class_boxes.size(0))).to(DEVICE)[class_images == this_image][ind]
            
            # Determine if detection is TP or FP
            if max_overlap.item() > 0.5:
                if t_object_diff[ind] == 0:  # Not difficult
                    if class_boxes_detected[original_ind] == 0:
                        TP[d] = 1
                        class_boxes_detected[original_ind] = 1
                    else:
                        FP[d] = 1
                # If object is difficult, ignore it
            else:
                FP[d] = 1
        
        # Calculate cumulative TP and FP
        cumul_TP = torch.cumsum(TP, dim=0)
        cumul_FP = torch.cumsum(FP, dim=0)
        
        # Calculate precision and recall
        cumul_precision = cumul_TP / (cumul_TP + cumul_FP + 1e-10)
        cumul_recall = cumul_TP / (easy_class_obj + 1e-10)
        
        # Calculate average precision
        recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).to(DEVICE)
        precisions = torch.zeros(len(recall_thresholds), dtype=torch.float).to(DEVICE)
        
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        
        # Store AP for this class
        AP[c - 1] = precisions.mean()
    
    # Calculate mAP
    mean_average_precision = AP.mean().item()
    
    # Create dictionary of class AP values
    average_precisions = {rev_label_map[c + 1]: v.item() for c, v in enumerate(AP)}
    
    return average_precisions, mean_average_precision


def transform(image, boxes , labels , split) :
    """
    Apply the trnasfomration above 

    Args : 
        image : a PIL image 
        boxes : a set of bbox 
        labels :   list of integrs 
        diffic : don't care 
        split  : Train or Test 

    Return : 
        transformed image , transformed bboxe , transformed labels, transformed diffic 
    """


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image  
    new_boxes = boxes 
    new_labels = labels

    if split.upper() == "TRAIN": 
        new_image = photometric_distort(new_image)


        new_image = FT.to_tensor(new_image)

        # crop the image 
        new_image , new_boxes , new_labels= random_crop(new_image , new_boxes , new_labels)

        new_image = FT.to_pil_image(new_image)

        # here i remove some augmentation ...Etc 

    new_image , new_boxes = resize(new_image , new_boxes , dims=(300,300))

    new_image = FT.to_tensor(new_image)

    new_image = FT.normalize(new_image, mean=mean , std=std)

    return new_image , new_boxes  , new_labels 

def random_crop(image , boxes , labels) : 
    """
        perform random crop to the image like in the paper 
    """
    return image , boxes , labels


def resize(image, boxes, dims=(300, 300), percent_cords=True):
    """
    Resize for the SSD300 sizes, resize to (300,300)
    """
    new_image = FT.resize(image, dims)
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims
    if percent_cords:
        mul_dims = torch.FloatTensor([dims[0], dims[1], dims[0], dims[1]]).unsqueeze(0)
        new_boxes *= mul_dims
    return new_image, new_boxes


def photometric_distort(image) : 
    """
        Distort brightness , contrast , saturation ...Etc
    """
    new_image = image 

    distorations= [FT.adjust_brightness , 
                   FT.adjust_contrast]
    random.shuffle(distorations)

    for d in distorations : 
        if random.random() < 0.5 : 
            adjust_factor = random.uniform(0.5 , 1.5)

            new_image = d(new_image , adjust_factor)

    return new_image



