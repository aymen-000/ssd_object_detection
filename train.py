import os 
import torch 
import time 
import torch.backends.cudnn as cudnn 
from model import  * 
from utils import *
from configuration import *
from torch.utils.data import DataLoader , Dataset
from torch.optim import SGD
from data import * 
cudnn.benchmark  = True 


def main() : 
    """
     Trainig the model 
    """


    if CHECKPOINTS is None : 
        start_epoch = 0 
        model = SSD300(n_classes=N_CLASSES)
        biases =[]
        weights= []
        for  param_name , paraem in model.named_parameters() :
            if param_name.endswith(".bias") : 
                biases.append(paraem)
            else : 
                weights.append(paraem)

            
        optim = SGD(params=[{'params': biases, 'lr': 2 * LR}, {'params': weights}],
                                    lr=LR, momentum=MOUMENTUM, weight_decay=WEIGHT_DACAY)
        
    else : 
        start_epoch, model, optimizer = load_model_pretrained_params(
            model=SSD300(n_classes=N_CLASSES),
            weight_file=CHECKPOINTS,
            optimizer=None,  # Create new optimizer or pass initial optimizer
            device=device
        )
    # Move to default device
    model = model.to(device)
    # CALCULATE LOSS
    criterion = BoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    # WOKING WITH DATA 

    data = LoadData() #  i should fill it 

    train_data_loader = DataLoader(
        data , 
        BATCH_SIZE , 
        shuffle=True , 
        num_workers=WORKERS , 
        collate_fn=data.collate_fn() , 
        pin_memory=True 
    )

    epochs =ITERS // (len(train_data_loader) // BATCH_SIZE)
    decay_lr_at = [it // (len(train_data_loader) // BATCH_SIZE) for it in DECAY_LR  ]


    # training loop 
    for epoch in range(epochs) : 
        if epoch in decay_lr_at : 
            # decay learning rate 
            decay_lr(optim ,DACAY_LR_COEFF)

        # One epoch's training
        train(train_loader=train_data_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoints(model=model , optimizer=optimizer , epoch=epoch)


def train(train_loader , model , criterion, optimizer , epoch) : 
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    for batch , (images ,bbox ,  labels, _) in enumerate(train_loader) : 
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

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
        if batch % PRINT_FRAQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored




