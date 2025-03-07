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
    

    





