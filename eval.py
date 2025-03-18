from tqdm import tqdm
from pprint import PrettyPrinter
from data import *
from utils import *
from configuration import *
from torch.utils.data import DataLoader
from model import SSD300

# good printing
pp = PrettyPrinter()

# parameters
# load the model to do inference
checkpoint = torch.load(CHECKPOINTS)
model = checkpoint["model"]
model: SSD300 = model.to(DEVICE)
# eval mode
model.eval()
data = LoadData()
# test data
test_data = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data.collate_fn,
    num_workers=WORKERS,
    pin_memory=True
)

def eval(test_data, model: SSD300):
    """
    Evaluation function
    Args:
        test_data: test dataLoader
        model: model to do inference
    """
    det_boxes = list()
    det_labels = list()
    det_scores =list()
    true_boxes =  list()
    true_labels = list()
    true_difficulties = list()
    
    with torch.no_grad():
        for batch, (images, labels, boxes, difficulties) in enumerate(tqdm(test_data)):
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
            difficulties = [d.to(DEVICE) for d in difficulties]
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
    
    # calculate mAP (mean average precision)
    APs, mAP = calc_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    pp.pprint(APs)
    print(f"\nMean Average Precision (mAP): {mAP:.3f}")

if __name__ == '__main__':
    eval(test_data, model)