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

    return tensor[slices]


