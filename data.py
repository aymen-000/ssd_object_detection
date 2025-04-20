import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import transform
import os

class AminiCocoaDataset(Dataset):
    """
    A PyTorch Dataset class that accepts a pre-loaded DataFrame instead of reading CSVs.
    """
    def __init__(self, df, image_root, split="train", keep_difficult=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing all image annotations.
            image_root (str): Root directory where images are stored.
            split (str): Either 'train' or 'test'.
            keep_difficult (bool): Unused here but can be kept for compatibility.
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.split = split.lower()
        assert self.split in ["train", "test" , "val"]
        self.keep_difficult = keep_difficult

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        image_name = row["Image_ID"]
        image_path = os.path.join(self.image_root, image_name)
        image = Image.open(image_path).convert("RGB")
        
        # Normalize bounding boxes
        boxes = [row["xmin"] , row["ymin"] , row["xmax"], row["ymax"] ]
        boxes = torch.FloatTensor(boxes)

        label = torch.LongTensor([row["class_id"]])
        
        # Apply transformation
        image, boxes, label = transform(image, boxes, label, split=self.split)
        
        return image, boxes, label

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        images = torch.stack(images, dim=0)
        return images, boxes, labels