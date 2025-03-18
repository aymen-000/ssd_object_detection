import unittest
import torch
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import calc_mAP  , find_jaccard_overlap
from configuration import * 

class TestCalcMAP(unittest.TestCase) : 
    def setUp(self):
        """
            Set Up test data
        """
        self.label_map = {"background" : 0 , "person" : 1 , "car" : 2 , "dog" : 3 }
        self.rev_label_map = {v : k for (v , k) in zip(self.label_map.values() , self.label_map.keys())}
        self.true_boxes = [
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], device=DEVICE),  # Image 1
            torch.tensor([[0.2, 0.3, 0.4, 0.5]], device=DEVICE)  # Image 2
        ]
        
        # Ground truth labels
        self.true_labels = [
            torch.tensor([1, 2], device=DEVICE),  # Image 1: person, car
            torch.tensor([3], device=DEVICE)  # Image 2: dog
        ]
        
        # All objects are easy (not difficult)
        self.true_difficulties = [
            torch.tensor([0, 0], device=DEVICE),
            torch.tensor([0], device=DEVICE)
        ]
        
        # Detected boxes (3 in image 1, 2 in image 2)
        self.det_boxes = [
            torch.tensor([[0.12, 0.22, 0.32, 0.42], [0.52, 0.62, 0.72, 0.82], [0.1, 0.1, 0.2, 0.2]], device=DEVICE),
            torch.tensor([[0.21, 0.31, 0.41, 0.51], [0.6, 0.6, 0.7, 0.7]], device=DEVICE)
        ]
        
        # Detected labels
        self.det_labels = [
            torch.tensor([1, 2, 1], device=DEVICE),  # Image 1: person, car, person
            torch.tensor([3, 2], device=DEVICE)  # Image 2: dog, car (false positive)
        ]
        
        # Detection confidence scores
        self.det_scores = [
            torch.tensor([0.9, 0.8, 0.7], device=DEVICE),
            torch.tensor([0.95, 0.6], device=DEVICE)
        ]
        self.calc_mAP = calc_mAP

    def test_perfect_detection(self) : 
        """
            Test with perfect detection (high IOU , correct labels)
        """
        global find_jaccard_overlap
        orig_jarcard = find_jaccard_overlap

        find_jaccard_overlap = self.perfect_iou

        try : 
            average_precisions, mean_average_precision = self.calc_mAP(
                self.det_boxes, self.det_labels, self.det_scores,
                self.true_boxes, self.true_labels, self.true_difficulties,
                self.label_map, self.rev_label_map
            )
            self.assertGreater(mean_average_precision , 0.9)
            self.assertEqual(len(average_precisions) , 3)
        finally : 
            find_jaccard_overlap = orig_jarcard
        
    def test_all_false_positives(self):
        """
        Test with only false positive detections (wrong labels)
        """

        wrong_det_labels = [
            torch.tensor([2, 3, 3], device=DEVICE),  # All wrong labels
            torch.tensor([1, 1], device=DEVICE)  # All wrong labels
        ]
        

        global find_jaccard_overlap
        orig_jaccard = find_jaccard_overlap
        find_jaccard_overlap = self.low_iou
        try:
            average_precisions, mean_average_precision = self.calc_mAP(
                self.det_boxes, wrong_det_labels, self.det_scores,
                self.true_boxes, self.true_labels, self.true_difficulties,
                self.label_map, self.rev_label_map
            )
            

            self.assertLess(mean_average_precision, 0.1)
        finally:
            find_jaccard_overlap = orig_jaccard       


    def perfect_iou(self ,set1 , set2) : 
        n1 = set1.size(0)
        n2 = set2.size(0)
        iou_matrix = torch.ones((n1, n2), device=DEVICE) * 0.9
        return iou_matrix
    

    def low_iou(self , set_1 , set_2) : 
        n1 = set_1.size(0)
        n2 = set_2.size(0)
        iou_matrix = torch.ones((n1, n2), device=DEVICE) * 0.1  # Below threshold
        return iou_matrix
    
    
if __name__ == "__main__" : 
    unittest.main()