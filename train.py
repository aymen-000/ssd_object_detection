import os 
import torch 
import time 
import argparse
import torch.backends.cudnn as cudnn 
from model import * 
from utils import *
from configuration import *
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from data import * 
from tqdm import tqdm
import pandas as pd 
import numpy as np
cudnn.benchmark = True 

def parse_args():
    """
    Parse command line arguments for training
    """
    parser = argparse.ArgumentParser(description='SSD300 Training')
    
    # Dataset parameters
    parser.add_argument('--data_folder', required=True, help='Path to the data folder')
    parser.add_argument('--labels_folder', required=True, help='Path to the labels folder')
    
    # Pretrained weights parameters
    parser.add_argument('--pretrained_weights', required=True, help='Path to pretrained SSD300 weights')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the model if specified')
    
    return parser.parse_args()

def load_pretrained_ssd(model, weights_path):
    """
    Load pretrained SSD300 weights
    
    Args:
        model: SSD300 model instance
        weights_path: Path to pretrained weights file
        
    Returns:
        model: Model with loaded weights
    """
    print(f"Loading pretrained weights from {weights_path}")
    
    # Load the weights
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    
    # Check if it's a state_dict or a full checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    
    # Load weights into model
    model.load_state_dict(state_dict, strict=False)

    print("Model weights loaded with success")
    return model

def split_data(df, val_ratio=0.2, seed=42):
    """
    Split dataframe into training and validation sets
    
    Args:
        df: DataFrame containing the data
        val_ratio: Ratio of validation data
        seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df: Split dataframes
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get unique image IDs to ensure we split by images, not by annotations
    unique_images = df["Image_ID"].unique()
    n_val = int(len(unique_images) * val_ratio)
    
    # Randomly select images for validation
    val_images = np.random.choice(unique_images, size=n_val, replace=False)
    
    # Split dataframe accordingly
    train_df = df[~df["Image_ID"].isin(val_images)]
    val_df = df[df["Image_ID"].isin(val_images)]
    
    return train_df, val_df

def main():
    """
    Training or evaluating the model with pretrained weights
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize model
    model = SSD300(n_classes=N_CLASSES)
    
    # Load pretrained weights
    model = load_pretrained_ssd(model, args.pretrained_weights)
    
    # Move to device
    model = model.to(DEVICE)
    
    # If fine-tuning, set up optimizer
    if args.fine_tune:
        # Prepare optimizer
        biases = []
        weights = []
        for param_name, param in model.named_parameters():
            if param_name.endswith(".bias"):
                biases.append(param)
            else:
                weights.append(param)
                
        optimizer = SGD(params=[{'params': biases, 'lr': 2 * LR * 0.1}, {'params': weights}],
                        lr=LR * 0.1, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        
        # Calculate loss 
        criterion = BoxLoss(priors=model.priors).to(DEVICE)
        
        # Working with data 
        df = pd.read_csv(os.path.join(args.labels_folder, "Train.csv"))    
        train_df, val_df = split_data(df)
        
        valid_data = AminiCocoaDataset(args.data_folder,  df=val_df, split="val")
        train_data = AminiCocoaDataset(args.data_folder, df=train_df, split="train")
        
        train_data_loader = DataLoader(
            train_data, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=WORKERS, 
            collate_fn=train_data.collate_fn, 
            pin_memory=True 
        )
        
        valid_data_loader = DataLoader(
            valid_data, 
            batch_size=32, 
            shuffle=False, 
            num_workers=WORKERS, 
            collate_fn=valid_data.collate_fn,
            pin_memory=True
        )

        # For fine-tuning, use fewer iterations
        fine_tune_iters = ITERS // 10
        epochs = fine_tune_iters // (len(train_data_loader) // BATCH_SIZE)
        decay_lr_at = [it // (len(train_data_loader) // BATCH_SIZE) for it in [fine_tune_iters//2, fine_tune_iters*3//4]]

        # Fine-tuning loop 
        print(f"Fine-tuning for {epochs} epochs")
        for epoch in range(epochs): 
            if epoch in decay_lr_at: 
                # decay learning rate 
                decay_lr(optimizer, DECAY_LR_COEFF)

            # One epoch's training
            train(train_loader=train_data_loader,
                val_loader=valid_data_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch)

            # Save checkpoint
            save_checkpoints(model=model, optimizer=optimizer, epoch=epoch, filename=f'fine_tuned_ssd300_epoch_{epoch}.pth')
    
    else:
        # Evaluation mode
        model.eval()
        print("Model loaded with pretrained weights and set to evaluation mode")
        
        # You can now use the model for inference/detection
        print("Ready for inference")


def train(train_loader, val_loader, model, criterion, optimizer, epoch): 
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    for batch_idx, (images, boxes, labels) in tqdm(enumerate(train_loader)): 
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(DEVICE)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(DEVICE) for b in boxes]
        labels = [l.to(DEVICE) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if GRAD_CLIP is not None:
            clip_gradient(optimizer, GRAD_CLIP)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if batch_idx % PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, batch_idx, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    
    # Validate after each epoch
    validate(val_loader, model, criterion)
    
    # Save model
    torch.save(model.state_dict(), f"SSD300_finetuned_epoch_{epoch}.pth")
    
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def validate(val_loader, model, criterion):
    """
    Validate the model on the validation set
    
    Args:
        val_loader: Validation data loader
        model: SSD300 model
        criterion: Loss function
    """
    model.eval()  # eval mode disables dropout
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    start = time.time()
    
    # No gradients needed for validation
    with torch.no_grad():
        for i, (images, boxes, labels) in enumerate(val_loader):
            # Move to default device
            images = images.to(DEVICE)
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]
            
            # Forward prop
            predicted_locs, predicted_scores = model(images)
            
            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)
            
            start = time.time()
            
            # Print status
            if i % PRINT_FREQ == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
    
    print('\n * VALIDATION LOSS - {loss.avg:.3f}\n'.format(loss=losses))
    
    model.train()  # Back to training mode
    
    return losses.avg


if __name__ == "__main__":
    main()