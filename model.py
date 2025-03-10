from torch import nn 
from utils import * 
from math import  * 
from itertools import product 
import torchvision
import torch.nn.functional as F # for manual operations 
import torch
from torchvision.models import vgg16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## VGG MODEL 
class VGGBaseModel(nn.Module) : 
    """
        VGG as base architacture
    """

    def __init__(self, *args, **kwargs):
        super(VGGBaseModel).__init__(*args, **kwargs)

        ## Conv layers in VGG
        self.conv1_1 = nn.Conv2d(3,64 , kernel_size=3 , padding=1) 
        self.conv1_2 = nn.Conv2d(64 , 64 , kernel_size=3 , padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2 , stride=2)
        
        # second block 
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block 
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #fourth block 
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fifth block 
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool_5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


        # Replacements for FC6 and FC7 in VGG16
        self.conv_6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        self.conv_7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # load pretrinaed model 
        self.load_pretrained_layer()
    def forward(self , image): 
        """
            Forawrd (predection) 

            Args : 
                image : a tensor of dimension (batch_size , 3,300 , 300)
            
            Return : 
                feature maps conv4_3 and conv_7
        """
        assert image.shape[1:] == (3,300,300)
        x = F.relu(self.conv1_1(image))
        x = F.relu(self.conv1_2(x))
        x = self.pool_1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool_2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool_3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv_4_3_feat = x  # (N, 512, 38, 38)
        x = self.pool_4(x)


        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool_5(x)

        x = F.relu(self.conv6(x))

        conv_7_feat = F.relu(self.conv7(x)) # (N, 1024, 19, 19)

        return conv_4_3_feat , conv_7_feat


    def load_pretrained_layer(self) : 
        """
            We are going to use VGG-16  pretrained on ImageNet task , We copy it's parmas to our network 
            However, the original VGG-16 does not contain the conv6 and con7 layers.
            Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation
        """
        # current model state 
        state_dict = self.state_dict()
        param_names = list(state_dict.keys()) #  return a list of layers names [conv_1_1...etc] 

        # pretrinaed VGG base model 
        pretrained_state_dict = vgg16(pretrained=True).state_dict()
        pretrained_params_names = list(pretrained_state_dict.keys())

        for i , param in enumerate(param_names[:4]) : 
            # execluding  conv6 and conv7
            state_dict[param] = pretrained_state_dict[pretrained_params_names[i]]

        # convert our fc6 and fc7 to convulution 
        conv_fc6_weights = pretrained_state_dict["classifier.0.weight"].view(4096 , 412 , 7, 7)
        conv_f6_bias = pretrained_state_dict['classifier.0.bias']

        state_dict["conv6.weight"] = decimate(conv_fc6_weights , m= [4 , None , 3 , 3]) # (1024, 512, 3, 3)
        state_dict["conv6.bias"] = decimate (conv_f6_bias , m=[4]) # 1024

        conv_fc7_weights  = pretrained_state_dict["classifier.3.weight"]
        conv_fc7_bias = pretrained_state_dict["classifier.3.bias"]
        
        state_dict["conv7.weight"] = decimate(conv_fc7_weights , m=[4 , 4 , None , None]) # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate (conv_fc7_bias, m=[4])  # (1024)

        self.load_state_dict(state_dict)

        print("\n Loaded base model weight with sucess \n")


## add more architactures here future work 

class AuxiliaryConvlutions(nn.Module): 
    """
        More convulutions to produce higher-level featur maps . 
    """ 
    def __init__(self, *args, **kwargs):
        super(AuxiliaryConvlutions).__init__(*args, **kwargs) 

        # make them on top of VGG base model 
        self.con
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) 

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        # init our conv parms 
        self.init_convs()

    def init_convs(self) : 
        """
            Initialize convulution parmaeters 
        """
        for c in self.children() : 
            if isinstance(c, nn.Conv2d) : 
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias , 0.)


    def forward(self, conv7_feats):
        """
        Forward propagation.
        Args 

            conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
    
        Output 
            higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        x = F.relu(self.conv8_1(conv7_feats)) 
        x = F.relu(self.conv8_2(x)) 
        conv8_2_feats = x

        x = F.relu(self.conv9_1(x))
        x = F.relu(self.conv9_2(x)) 
        conv9_2_feats = x  # (N, 256, 5, 5)

        x = F.relu(self.conv10_1(x)) 
        x = F.relu(self.conv10_2(x))  
        conv10_2_feats = x  # (N, 256, 3, 3)

        x = F.relu(self.conv11_1(x)) 
        conv11_2_feats = F.relu(self.conv11_2(x)) 

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats



class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict bbox and scores for higher and lower feature maps.
    bbox predicted as offset.
    score represents the score of each object class in each bbox which is located.
    Note: higher score of "background" => no object
    """
    def __init__(self, n_classes, **kwargs):
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        # define the number of bboxes in each feature map
        self.n_boxes = {
            "conv4_3": 4,  # means we used 4 different aspect ratios ...etc
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
            "conv11_2": 4
        }
        
        # Input channels for each feature map
        self.input_channels = {
            "conv4_3": 512,
            "conv7": 1024,
            "conv8_2": 512,
            "conv9_2": 256,
            "conv10_2": 256,
            "conv11_2": 256
        }
        
        # Create all bbox and class prediction convolutions using get_bbox_cls
        self.loc_conv4_3, self.cl_conv4_3 = self.get_bbox_cls("conv4_3", self.input_channels["conv4_3"])
        self.loc_conv7, self.cl_conv7 = self.get_bbox_cls("conv7", self.input_channels["conv7"])
        self.loc_conv8_2, self.cl_conv8_2 = self.get_bbox_cls("conv8_2", self.input_channels["conv8_2"])
        self.loc_conv9_2, self.cl_conv9_2 = self.get_bbox_cls("conv9_2", self.input_channels["conv9_2"])
        self.loc_conv10_2, self.cl_conv10_2 = self.get_bbox_cls("conv10_2", self.input_channels["conv10_2"])
        self.loc_conv11_2, self.cl_conv11_2 = self.get_bbox_cls("conv11_2", self.input_channels["conv11_2"])
    
    def get_bbox_cls(self, feat_name, input_channels):
        """
        Helper function to create bbox and class prediction convolutions
        
        Args:
            feat_name: Feature map identifier (e.g., 'conv4_3')
            input_channels: Number of input channels
            
        Returns:
            tuple: (bbox_predictor, class_predictor) convolution layers
        """
        bbox = nn.Conv2d(input_channels, self.n_boxes[feat_name] * 4, kernel_size=3, padding=1)
        cls = nn.Conv2d(input_channels, self.n_boxes[feat_name] * self.n_classes, kernel_size=3, padding=1)
        return bbox, cls
    
    def reshape_conv_output(self, output, batch_size, n_values):
        """
        Reshape convolutional output for detection processing
        
        Args:
            output: Conv output tensor of shape (batch_size, channels, height, width)
            batch_size: Batch size
            n_values: Number of values per box (4 for locations, n_classes for class scores)
            
        Returns:
            Reshaped tensor of shape (batch_size, height*width*n_boxes, n_values)
        """
        # Permute dimensions to (batch_size, height, width, channels)
        output = output.permute(0, 2, 3, 1).contiguous()
        # Reshape to (batch_size, height*width*n_boxes, n_values)
        output = output.view(batch_size, -1, n_values)
        return output
    
    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.
        
        Args:
            conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, H_4, W_4)
            conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, H_7, W_7)
            conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, H_8, W_8)
            conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, H_9, W_9)
            conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, H_10, W_10)
            conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, H_11, W_11)
            
        Returns:
            all_locs, all_classes encoded as (N, n_boxes_total, 4) and (N, n_boxes_total, n_classes)
        """
        batch_size = conv4_3_feats.size(0)
        
        # Predict bounding box coordinates
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        
        # Reshape location predictions using our function
        l_conv4_3 = self.reshape_conv_output(l_conv4_3, batch_size, 4)  # (N, 5776, 4)
        l_conv7 = self.reshape_conv_output(l_conv7, batch_size, 4)  # (N, 2166, 4)
        l_conv8_2 = self.reshape_conv_output(l_conv8_2, batch_size, 4)  # (N, 600, 4)
        l_conv9_2 = self.reshape_conv_output(l_conv9_2, batch_size, 4)  # (N, 150, 4)
        l_conv10_2 = self.reshape_conv_output(l_conv10_2, batch_size, 4)  # (N, 36, 4)
        l_conv11_2 = self.reshape_conv_output(l_conv11_2, batch_size, 4)  # (N, 4, 4)
        
        # Predict class scores
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        
        # Reshape class predictions using our function
        c_conv4_3 = self.reshape_conv_output(c_conv4_3, batch_size, self.n_classes)  # (N, 5776, n_classes)
        c_conv7 = self.reshape_conv_output(c_conv7, batch_size, self.n_classes)  # (N, 2166, n_classes)
        c_conv8_2 = self.reshape_conv_output(c_conv8_2, batch_size, self.n_classes)  # (N, 600, n_classes)
        c_conv9_2 = self.reshape_conv_output(c_conv9_2, batch_size, self.n_classes)  # (N, 150, n_classes)
        c_conv10_2 = self.reshape_conv_output(c_conv10_2, batch_size, self.n_classes)  # (N, 36, n_classes)
        c_conv11_2 = self.reshape_conv_output(c_conv11_2, batch_size, self.n_classes)  # (N, 4, n_classes)
        
        # Concatenate all predictions
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)
        
        return locs, classes_scores

    

class SSD300(nn.Module) : 
    """
        SSD300 network 

    """

    def __init__(self, n_classes, **kwargs):
        super(SSD300).__init__()
        
        self.n_classes = n_classes

        self.model =VGGBaseModel()
        self.aux_conv = AuxiliaryConvlutions()
        self.pred_convs = PredictionConvolutions(n_classes=n_classes)

        self.scal_f = nn.Parameter(torch.FloatTensor(1,412 , 1, 1))
        nn.init.constant_(self.scal_f , 20)
        self.priors = self.create_prior_boxes()
    def forward(self , image)  :
        """
            Forward Propagation 

            Args : 
                image : (N, 3 , 300 , 300)

            Return : 
                8732 locations and class scores 
        """

        # Run VGG16 on the model 
        conv4_3_feats , conv7_feats = self.model(image) # return (N , 512 , 38 , 38) , (N ,1024 , 19 , 19)
        mean = conv4_3_feats.sum(dim=1, keepdim=True) / 512  # Compute mean 
        norm = (conv4_3_feats - mean).pow(2).sum(dim=1, keepdim=True) / 512  # Compute variance
        conv4_3_feats = ((conv4_3_feats - mean) / (norm.sqrt() + 1e-6)) * self.scal_f  # Normalize


        # more more featur extraction 
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_conv(conv7_feats)

        # Run predection 
        locs , cls_scores = self.pred_convs(conv4_3_feats , conv7_feats , conv8_2_feats , conv9_2_feats , conv10_2_feats, conv11_2_feats) # we will get (N , nboxes , 4) , (N , nboxes , nclasses)

        return locs , cls_scores


    def create_prior_boxes(self) : 
        """
            Create the 8732 prior (deafult) boxes for SSD300
        """


        fmap_dims = {
            'conv4_3' : 38 , 
            'conv7' : 19 , 
            'conv8_2' : 10, 
            'conv9_2' : 5 , 
            'conv10_2' : 3, 
            'conv11_2' : 1
        }

        obj_scales = {
            'conv4_3' : 0.1 , # define the scale to detect objects (various sizes)
            'conv7' : 0.2 , 
            'conv8_2' : 0.375, 
            'conv9_2' : 0.55 , 
            'conv10_2' : 0.725, 
            'conv11_2' : 0.9 
        }
        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}


        prior_boxes = []

        for k , fmap in list(fmap_dims.keys()) : 
            for i in range(fmap_dims[fmap]) : 
                for j in range(fmap_dims[fmap]) : 
                    cx = (j+0.5) / fmap_dims[fmap]
                    cy = (i+ 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap] : 
                        prior_boxes.append([cx , cy , obj_scales[fmap]*sqrt(ratio) , obj_scales[fmap]/ sqrt(ratio)])

                        if ratio == 1 : 
                            try : 
                                scale =sqrt(obj_scales[fmap]*obj_scales[fmap[k+1]])
                            except IndexError : 
                                scale = 1 
                            prior_boxes.append([cx , cy , scale , scale])

        return torch.FloatTensor(prior_boxes).to(device=device)



    def detect_objects(self , pred_locs , pred_scores , threshold , max_overlap , k) : 
        """
            Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

            For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

            Args : 

            Retrurn  : 

        """

        batch_size = pred_locs.size(0)
        
        pred_scores = F.softmax(pred_scores , dim=2) 

        final_img_bbox = []
        final_img_labels = []
        final_img_scores = []

        for i in range(batch_size) : 
            
            locs = cxcy_to_xy(
                pred_to_boxes(pred_locs[i] ,self.priors) # model correct predections
            ) # turn to x_min , x_max , y_min , y_max

            images_boxes = []
            images_labels = []
            images_scores = []

            max_scr , best_label = pred_scores[i].max(dim=1)

            for c in range(1 , self.n_classes) : 
                class_scores = pred_scores[i][: , c] 
                score_abover_threshold = class_scores > threshold 
                n_above_threshold = score_abover_threshold.sum().item()
                if n_above_threshold == 0 : 
                    continue
                class_scores = class_scores[score_abover_threshold]
 








