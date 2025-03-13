import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import VGGBaseModel, AuxiliaryConvolutions, PredictionConvolutions , SSD300

class TestObjectDetectionModels(unittest.TestCase):
    """Combined test class for object detection model components"""
    
    def setUp(self):
        """Common setup for all tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.n_classes = 21  # 20 object classes + 1 background
    
    # VGGBaseModel Tests
    def test_vgg_output_shape(self):
        """Test if VGGBaseModel outputs feature maps with expected shapes"""
        model = VGGBaseModel()
        model.eval()
        input_tensor = torch.rand((self.batch_size, 3, 300, 300))
        
        # Output of the model
        conv4_3_feat, conv7_feat = model(input_tensor)
        expected_output_7 = (self.batch_size, 1024, 19, 19)
        expected_output_4_3 = (self.batch_size, 512, 38, 38)
        
        self.assertEqual(conv4_3_feat.shape, expected_output_4_3, "conv4_3 shapes not matched")
        self.assertEqual(conv7_feat.shape, expected_output_7, "conv7 shapes mismatch")
    
    def test_vgg_forward_pass(self):
        """Ensure that the VGGBaseModel runs inference correctly"""
        model = VGGBaseModel()
        model.eval()
        input_tensor = torch.rand((self.batch_size, 3, 300, 300))
        
        try:
            _ = model(input_tensor)
        except Exception as e:
            self.fail(f"Forward pass with VGGBaseModel failed with error: {e}")
    
    def test_vgg_require_grad(self):
        """Ensure that the VGGBaseModel parameters require gradient"""
        model = VGGBaseModel()
        
        for param in model.parameters():
            self.assertTrue(param.requires_grad, "Some VGGBaseModel params don't require gradient")
    
    # AuxiliaryConvolutions Tests
    def test_aux_conv_output_shapes(self):
        """Test if AuxiliaryConvolutions outputs feature maps with expected shapes"""
        model = AuxiliaryConvolutions()
        model.eval()
        # Create mock conv7_feats with shape (batch_size, 1024, 19, 19)
        conv7_feats = torch.rand((self.batch_size, 1024, 19, 19))
        
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = model(conv7_feats)
        
        # Expected output shapes
        expected_shape_conv8_2 = (self.batch_size, 512, 10, 10)
        expected_shape_conv9_2 = (self.batch_size, 256, 5, 5)
        expected_shape_conv10_2 = (self.batch_size, 256, 3, 3)
        expected_shape_conv11_2 = (self.batch_size, 256, 1, 1)
        
        self.assertEqual(conv8_2_feats.shape, expected_shape_conv8_2, "conv8_2 shapes don't match")
        self.assertEqual(conv9_2_feats.shape, expected_shape_conv9_2, "conv9_2 shapes don't match")
        self.assertEqual(conv10_2_feats.shape, expected_shape_conv10_2, "conv10_2 shapes don't match")
        self.assertEqual(conv11_2_feats.shape, expected_shape_conv11_2, "conv11_2 shapes don't match")
    
    def test_aux_conv_forward_pass(self):
        """Ensure that the AuxiliaryConvolutions model runs inference correctly"""
        model = AuxiliaryConvolutions()
        model.eval()
        conv7_feats = torch.rand((self.batch_size, 1024, 19, 19))
        
        try:
            _ = model(conv7_feats)
        except Exception as e:
            self.fail(f"Forward pass with AuxiliaryConvolutions failed with error: {e}")
    
    def test_aux_conv_require_grad(self):
        """Ensure that the AuxiliaryConvolutions parameters require gradient"""
        model = AuxiliaryConvolutions()
        
        for param in model.parameters():
            self.assertTrue(param.requires_grad, "Some AuxiliaryConvolutions params don't require gradient")
    
    def test_aux_conv_initialization(self):
        """Test if AuxiliaryConvolutions weights are initialized correctly"""
        model = AuxiliaryConvolutions()
        
        # Check if weights are initialized with Xavier uniform
        for module in model.children():
            if isinstance(module, nn.Conv2d):
                # Check if weights are not all zeros or ones
                self.assertFalse(torch.all(torch.eq(module.weight, 0.0)), 
                                "Weights should not be all zeros")
                self.assertFalse(torch.all(torch.eq(module.weight, 1.0)), 
                                "Weights should not be all ones")
                
                # Check if biases are initialized to zero
                self.assertTrue(torch.all(torch.eq(module.bias, 0.0)), 
                               "Biases should be initialized to zero")
    
    # PredictionConvolutions Tests
    def test_pred_conv_initialization(self):
        """Test if PredictionConvolutions initializes with correct parameters"""
        model = PredictionConvolutions(n_classes=self.n_classes)
        
        # Check if n_classes is set correctly
        self.assertEqual(model.n_classes, self.n_classes, "n_classes not set correctly")
        
        # Check if n_boxes is set correctly for each feature map
        expected_n_boxes = {
            "conv4_3": 4,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
            "conv11_2": 4
        }
        self.assertEqual(model.n_boxes, expected_n_boxes, "n_boxes not set correctly")
    
    def test_pred_conv_output_shapes(self):
        """Test if PredictionConvolutions outputs tensors with expected shapes"""
        model = PredictionConvolutions(n_classes=self.n_classes)
        model.eval()
        
        # Create mock feature maps
        conv4_3_feats = torch.rand((self.batch_size, 512, 38, 38))
        conv7_feats = torch.rand((self.batch_size, 1024, 19, 19))
        conv8_2_feats = torch.rand((self.batch_size, 512, 10, 10))
        conv9_2_feats = torch.rand((self.batch_size, 256, 5, 5))
        conv10_2_feats = torch.rand((self.batch_size, 256, 3, 3))
        conv11_2_feats = torch.rand((self.batch_size, 256, 1, 1))
        
        locs, classes_scores = model(
            conv4_3_feats, conv7_feats, conv8_2_feats, 
            conv9_2_feats, conv10_2_feats, conv11_2_feats
        )
        
        # Calculate expected total number of boxes
        # 4 boxes per position in conv4_3: 38×38×4 = 5776
        # 6 boxes per position in conv7: 19×19×6 = 2166
        # 6 boxes per position in conv8_2: 10×10×6 = 600
        # 6 boxes per position in conv9_2: 5×5×6 = 150
        # 4 boxes per position in conv10_2: 3×3×4 = 36
        # 4 boxes per position in conv11_2: 1×1×4 = 4
        # Total: 8732
        expected_total_boxes = 8732
        
        # Check shapes
        expected_locs_shape = (self.batch_size, expected_total_boxes, 4)
        expected_classes_shape = (self.batch_size, expected_total_boxes, self.n_classes)
        
        self.assertEqual(locs.shape, expected_locs_shape, "Locations shape doesn't match expected")
        self.assertEqual(classes_scores.shape, expected_classes_shape, "Classes scores shape doesn't match expected")
    
    def test_pred_conv_forward_pass(self):
        """Ensure that the PredictionConvolutions model runs inference correctly"""
        model = PredictionConvolutions(n_classes=self.n_classes)
        model.eval()
        
        # Create mock feature maps
        conv4_3_feats = torch.rand((self.batch_size, 512, 38, 38))
        conv7_feats = torch.rand((self.batch_size, 1024, 19, 19))
        conv8_2_feats = torch.rand((self.batch_size, 512, 10, 10))
        conv9_2_feats = torch.rand((self.batch_size, 256, 5, 5))
        conv10_2_feats = torch.rand((self.batch_size, 256, 3, 3))
        conv11_2_feats = torch.rand((self.batch_size, 256, 1, 1))
        
        try:
            locs, classes_scores = model(
                conv4_3_feats, conv7_feats, conv8_2_feats, 
                conv9_2_feats, conv10_2_feats, conv11_2_feats
            )
            
            # Check that outputs are not NaN or infinity
            self.assertFalse(torch.isnan(locs).any(), "Locations contain NaN values")
            self.assertFalse(torch.isinf(locs).any(), "Locations contain infinity values")
            self.assertFalse(torch.isnan(classes_scores).any(), "Class scores contain NaN values")
            self.assertFalse(torch.isinf(classes_scores).any(), "Class scores contain infinity values")
            
        except Exception as e:
            self.fail(f"Forward pass with PredictionConvolutions failed with error: {e}")
    
    def test_pred_conv_require_grad(self):
        """Ensure that the PredictionConvolutions parameters require gradient"""
        model = PredictionConvolutions(n_classes=self.n_classes)
        
        for param in model.parameters():
            self.assertTrue(param.requires_grad, "Some PredictionConvolutions params don't require gradient")
    
    # End-to-end Tests
    def test_integration_vgg_aux_conv(self):
        """Test integration between VGGBaseModel and AuxiliaryConvolutions"""
        vgg_model = VGGBaseModel()
        aux_conv_model = AuxiliaryConvolutions()
        
        input_tensor = torch.rand((self.batch_size, 3, 300, 300))
        
        try:
            # Forward pass through VGG
            _, conv7_feat = vgg_model(input_tensor)
            
            # Forward pass through AuxiliaryConvolutions
            conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = aux_conv_model(conv7_feat)
            
            # Verify shapes
            self.assertEqual(conv7_feat.shape, (self.batch_size, 1024, 19, 19), "Intermediate feature map shape incorrect")
            self.assertEqual(conv11_2_feats.shape, (self.batch_size, 256, 1, 1), "Final feature map shape incorrect")
            
        except Exception as e:
            self.fail(f"Integration test failed with error: {e}")
    
    def test_full_integration(self):
        """Test full integration of VGG, AuxiliaryConvolutions, and PredictionConvolutions"""
        vgg_model = VGGBaseModel()
        aux_conv_model = AuxiliaryConvolutions()
        pred_conv_model = PredictionConvolutions(n_classes=self.n_classes)
        
        input_tensor = torch.rand((self.batch_size, 3, 300, 300))
        
        try:
            # Forward pass through VGG
            conv4_3_feat, conv7_feat = vgg_model(input_tensor)
            
            # Forward pass through AuxiliaryConvolutions
            conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = aux_conv_model(conv7_feat)
            
            # Forward pass through PredictionConvolutions
            locs, classes_scores = pred_conv_model(
                conv4_3_feat, conv7_feat, conv8_2_feats, 
                conv9_2_feats, conv10_2_feats, conv11_2_feats
            )
            
            # Verify final output shapes
            expected_total_boxes = 8732
            self.assertEqual(locs.shape, (self.batch_size, expected_total_boxes, 4), 
                             "Final locations shape incorrect")
            self.assertEqual(classes_scores.shape, (self.batch_size, expected_total_boxes, self.n_classes), 
                             "Final class scores shape incorrect")
            
        except Exception as e:
            self.fail(f"Full integration test failed with error: {e}")

    def test_ssd_full_model(self) :
        ssd = SSD300(self.n_classes)
        input_tensor = torch.rand((self.batch_size, 3, 300, 300)) 
        ## test model forward 
        try : 
            locs , cls_scores = ssd(input_tensor) 
            self.assertEqual(locs.size() ,torch.Size([self.batch_size , 8732 , 4]))
            self.assertEqual(cls_scores.size() , torch.Size([self.batch_size , 8732 , self.n_classes]))
        except Exception as e : 
            self.fail(f"Failed in ssd forward with error {e}")
        
        
if __name__ == "__main__":
    unittest.main()