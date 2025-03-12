import unittest
import torch
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import BoxLoss
# Add the parent directory (SSD/) to sys.path

class TestBoxLoss(unittest.TestCase):
    
    def setUp(self):
        """Initialize loss function and mock data."""
        num_priors = 5  # Example number of priors
        num_classes = 3  # Example class count (background + 2 objects)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.priors = torch.rand((num_priors, 4)).to(self.device)
        self.loss_fn = BoxLoss(self.priors).to(self.device)

        self.predicted_locs = torch.rand((2, num_priors, 4)).to(self.device)  # Batch size 2
        self.predicted_scores = torch.rand((2, num_priors, num_classes)).to(self.device)
        self.boxes = [torch.rand((3, 4)).to(self.device) for _ in range(2)]  # 3 GT boxes per image
        self.labels = [torch.randint(0, num_classes, (3,)).to(self.device) for _ in range(2)]

    def test_loss_shape(self):
        """Test if the loss returns a scalar tensor."""
        loss = self.loss_fn(self.predicted_locs, self.predicted_scores, self.boxes, self.labels)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Should be a scalar

    def test_loss_value(self):
        """Ensure loss produces reasonable values (not NaN or infinite)."""
        loss = self.loss_fn(self.predicted_locs, self.predicted_scores, self.boxes, self.labels)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertGreater(loss.item(), 0)  # Loss should be positive

    def test_empty_boxes(self):
        """Test handling of an empty box list."""
        empty_boxes = [torch.empty((0, 4)).to(self.device) for _ in range(2)]
        empty_labels = [torch.empty((0,), dtype=torch.long).to(self.device) for _ in range(2)]
        
        loss = self.loss_fn(self.predicted_locs, self.predicted_scores, empty_boxes, empty_labels)

    

if __name__ == '__main__':
    unittest.main()
