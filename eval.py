from tqdm import tqdm
from pprint import PrettyPrinter
from data import *
from utils import *
from configuration import *
from torch.utils.data import DataLoader
from model import SSD300
import argparse
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='SSD300 Evaluation')
parser.add_argument('--data_folder', required=True, help='Path to the data folder')
parser.add_argument('--labels_file', required=True, help='Path to the labels folder')
parser.add_argument("--model_path", required=True, help="Path to state dict model")
parser.add_argument("--label_map" ,default="./labels.json" ,  required=True , help="add lable map in json format {class_id : class}")
args = parser.parse_args()

# Good printing
pp = PrettyPrinter()
# split data to get validation data 
# Load the data
df = pd.read_csv(args.labels_file)
labels_map , rev_labels_map = parse_json(args.label_map)

_, val_data = split_data(df)
data = AminiCocoaDataset(val_data, args.data_folder, split="val")

# Test data loader
test_data = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data.collate_fn,
    num_workers=WORKERS,
    pin_memory=True
)

# Initialize and load the model
model = SSD300(n_classes=N_CLASSES)
state_dict = torch.load(args.model_path)
model = state_dict_to_model(model, state_dict)
model = model.to(DEVICE)
model.eval()

def eval(test_data, model: SSD300):
    """
    Evaluation function
    Args:
        test_data: test dataLoader
        model: model to do inference
    """
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    
    with torch.no_grad():
        for batch, (images, boxes, labels ) in enumerate(tqdm(test_data)):
            images = images.to(DEVICE)
            pred_locs, pred_scores = model(images)
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                pred_locs=pred_locs,
                pred_scores=pred_scores,
                threshold=0.01,
                max_overlap=0.45,
                k=200
            )
            
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
    # Calculate mAP (mean average precision)
    APs, mAP = calc_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties , label_map=labels_map , rev_label_map=rev_labels_map)
    pp.pprint(APs)
    print(f"\nMean Average Precision (mAP): {mAP:.3f}")

if __name__ == '__main__':
    eval(test_data, model)