import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
from PIL import Image
from utils import transform
import pandas as pd

class AminiCocoaDataset(Dataset):
    """
    A pytorch class to be used in pytorch dataloader during training loop
    """
    def __init__(self, data_folder, labels_folder , split="train", keep_difficult=False):
        super().__init__()
        self.split = split.lower()
        assert self.split in ["train", "test"]
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        
        self.csv_train = os.path.join(labels_folder, "Train.csv")
        self.csv_test = os.path.join(labels_folder, "Test.csv")
        self.train_path = os.path.join(data_folder, 'images', 'train')
        self.test_path = os.path.join(data_folder, "images", "test")
        
        # Load the appropriate CSV based on split
        if self.split == "train":
            self.csv_file = self.csv_train
            self.images_path = self.train_path
        else:
            self.csv_file = self.csv_test
            self.images_path = self.test_path
            
        self.df = pd.read_csv(self.csv_file)

    def __getitem__(self, index):
        assert index < len(self.df)
        row = self.df.iloc[index]
        
        image_name = row["Image_ID"]
        image = Image.open(os.path.join(self.images_path, image_name))
        image = image.convert("RGB")
        
        # Read the bounding boxes in the image
        boxes = [row["ymin"], row["xmin"], row["ymax"], row["xmax"]]
        boxes = torch.FloatTensor(boxes)
        
        label = torch.LongTensor([row["class_id"]])
        
        # Apply transformation
        image, boxes, label = transform(image, boxes, label)
        
        return image, boxes, label

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        """
        We are going to handle all the cases of image containing one object or multiple
        
        Args:
            batch: a batch from __getitem__() method
            
        Return:
            a tensor of images, lists of varying-size tensors of bounding boxes, labels
        """
        images = []
        boxes = []
        labels = []
        
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        
        images = torch.stack(images, dim=0)
        
        return images, boxes, labels